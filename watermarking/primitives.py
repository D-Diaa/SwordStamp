"""Primitives adapted from original SemStamp's LSH and KMeans utilities."""

import os
import pickle
from pathlib import Path
from typing import Iterator, List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from nearpy.hashes import RandomBinaryProjections
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from kmeans_pytorch import *
from config.paths import segmentation_cache_tag
from segmentation import (
    DEFAULT_BACKEND,
    DEFAULT_SEMCUT_BATCH_SIZE,
    DEFAULT_SEMCUT_MAX_WORDS,
    DEFAULT_SEMCUT_WINDOW,
    DEFAULT_TYPE,
    Segmenter,
    segment,
)

# Shared acceptance-mask salt.
hash_key = 15485863

device = "cuda"
_rng = None


def _get_rng():
    global _rng
    if _rng is None:
        _rng = torch.Generator(device)
    return _rng


def extract_prompt_from_text(
    text,
    len_prompt,
    segmentation_type=DEFAULT_TYPE,
    segmentation_backend=DEFAULT_BACKEND,
):
    tokens = text.split()
    truncated = ' '.join(tokens[:len_prompt])
    units = segment(truncated, type=segmentation_type, backend=segmentation_backend)
    for unit in units:
        if len(unit.display.strip().split()) > 3:
            return unit.display.strip()
    return truncated + "."


class LSHModel:
    def __init__(self, device, batch_size, lsh_dim):
        self.dimension: int = -1
        self.device = device
        self.batch_size: int = batch_size
        self.lsh_dim: int = lsh_dim
        print("initializing random projection LSH model")
        self.hasher = RandomBinaryProjections(
            'rbp_perm', projection_count=self.lsh_dim, rand_seed=1234)

    def get_embeddings(self, sents: Iterator[str]) -> np.ndarray:
        """Embed sentence strings."""
        raise NotImplementedError()

    def get_hash(self, sents: Iterator[str], embeds=None) -> Iterator[str]:
        if embeds is None:
            embeds = self.get_embeddings(sents)
        hash_strs = [self.hasher.hash_vector(e)[0] for e in embeds]
        hash_ints = [int(s, 2) for s in hash_strs]
        return hash_ints


class SBERTLSHModel(LSHModel):
    def __init__(self, device, batch_size, lsh_dim, sbert_type='roberta', lsh_model_path=None, **kwargs):
        super(SBERTLSHModel, self).__init__(device, batch_size, lsh_dim)
        self.sbert_type = sbert_type
        self.dimension = 1024 if 'large' in self.sbert_type else 768

        print(f'loading SBERT {self.sbert_type} model...')
        if lsh_model_path is not None:
            self.embedder = SentenceTransformer(lsh_model_path)
            self.dimension = self.embedder.get_sentence_embedding_dimension()
        else:
            self.embedder = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v1")
        self.embedder = self.embedder.to(self.device)
        self.embedder.eval()

        self.hasher.reset(dim=self.dimension)

    def get_embeddings(self, sents: Iterator[str]) -> np.ndarray:
        all_embeddings = self.embedder.encode(
            sents, batch_size=self.batch_size)
        return np.stack(all_embeddings)


def cosine_distance_matrix(x, y):
    return F.cosine_similarity(
        x.view(x.size(0), 1, x.size(1))
        .expand(x.size(0), y.size(0), x.size(1))
        .contiguous()
        .view(-1, x.size(1)),
        y.expand(x.size(0), y.size(0), y.size(1)).flatten(end_dim=1),
    ).view(x.size(0), y.size(0))


def get_mask_from_seed(lsh_dim: int, accept_rate: float, seed: int, key: int = hash_key):
    n_bins = 2**lsh_dim
    n_accept = int(n_bins * accept_rate)
    # manual_seed requires an int64 value.
    rng = _get_rng()
    rng.manual_seed((key * int(seed)) & 0x7FFFFFFFFFFFFFFF)
    vocab_permutation = torch.randperm(n_bins, device=device, generator=rng)
    greenlist_ids = vocab_permutation[:n_accept]
    return greenlist_ids.to(device)


def compute_lsh_margins(lsh_model, sents, embeds=None, cutoff=None):
    """Return minimum absolute similarity to an LSH hyperplane."""
    if embeds is None:
        embeds = torch.tensor(lsh_model.get_embeddings(sents), device=device)
    elif not isinstance(embeds, torch.Tensor):
        embeds = torch.tensor(embeds, device=device)
    normals = torch.tensor(lsh_model.hasher.normals, device=device)
    if cutoff is not None:
        normals = normals[:cutoff]
    sims = cosine_distance_matrix(embeds, normals)
    return sims.abs().min(dim=1).values


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    if distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    X = X.float().to(device)
    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=-1)

    return choice_cluster.cpu()


def embedding_cache_path(
    dataset_path,
    embedder_path,
    segmentation_type=DEFAULT_TYPE,
    segmentation_backend=DEFAULT_BACKEND,
    semcut_max_words=DEFAULT_SEMCUT_MAX_WORDS,
    semcut_window=DEFAULT_SEMCUT_WINDOW,
):
    """Return the cache path for one embedding and segmentation policy."""
    seg_tag = segmentation_cache_tag(
        segmentation_type, segmentation_backend,
        semcut_max_words, semcut_window,
    )
    embed_tag = Path(embedder_path).name
    return os.path.join(dataset_path, f"embeds_{seg_tag}_{embed_tag}.pkl")


def embed_gen_list(
    dataset_path,
    embedder_path,
    encode_batch_size=32,
    num_gpus=None,
    segmentation_type=DEFAULT_TYPE,
    segmentation_backend=DEFAULT_BACKEND,
    semcut_max_words=DEFAULT_SEMCUT_MAX_WORDS,
    semcut_window=DEFAULT_SEMCUT_WINDOW,
    semcut_batch_size=DEFAULT_SEMCUT_BATCH_SIZE,
):
    """Embed every scoring unit in a dataset."""
    if num_gpus not in (None, 1):
        print(f"Ignoring num_gpus={num_gpus}; embedding precompute is single-process.")

    name = embedding_cache_path(
        dataset_path,
        embedder_path,
        segmentation_type,
        segmentation_backend,
        semcut_max_words,
        semcut_window,
    )
    if os.path.exists(name):
        print(f"Embeddings cache hit: {name}")
        return name

    dataset = load_from_disk(dataset_path)
    texts = dataset['text']
    embedder = SentenceTransformer(embedder_path, device=device).eval()
    segmenter = (
        Segmenter.from_sentence_transformer(
            segmentation_type, segmentation_backend, embedder, embedder_path,
            batch_size=semcut_batch_size,
            semcut_max_words=semcut_max_words,
            semcut_window=semcut_window,
        )
        if segmentation_type == "semspan"
        else Segmenter(segmentation_type, segmentation_backend)
    )

    # Match generation and detection granularity.
    all_segments: List[str] = []
    for text in tqdm(texts, desc="Segmenting"):
        units = segmenter.segment(text)
        all_segments.extend(unit.normalized for unit in units if unit.normalized.strip())

    all_embed_batches = []
    with tqdm(total=len(all_segments), desc="Encoding") as pbar:
        for i in range(0, len(all_segments), encode_batch_size):
            batch = all_segments[i:i + encode_batch_size]
            batch_embeds = embedder.encode(batch, convert_to_tensor=True)
            all_embed_batches.append(batch_embeds.detach().cpu())
            pbar.update(len(batch))

    stacked = torch.cat(all_embed_batches, dim=0)  # [N, D] — avoids 40GB list-of-tensors pickle
    with open(name, 'wb') as f:
        pickle.dump({'text': stacked}, f)

    print(f"Embeddings saved to {name} ({len(stacked)} segment embeddings from {len(texts)} documents)")
    return name


def get_cluster_mask(curr_cluster_id, k_dim, lmbd, key: int = hash_key):
    rng = _get_rng()
    rng.manual_seed((int(curr_cluster_id.item()) * key) & 0x7FFFFFFFFFFFFFFF)
    num_accept = int(k_dim * lmbd)
    mask = torch.randperm(k_dim, device=device, generator=rng)[:num_accept]
    return mask.to(device)


def compute_kmeans_margins(texts, embedder, cluster_centers):
    """Return nearest-cluster margins and IDs."""
    gen_embeds = embedder.encode(texts, convert_to_tensor=True)
    if gen_embeds.dim() == 1:
        gen_embeds = gen_embeds.unsqueeze(0)
    centers_t = cluster_centers if isinstance(cluster_centers, torch.Tensor) else torch.tensor(np.array(cluster_centers))
    dis = pairwise_cosine(gen_embeds, centers_t, device=device)
    if dis.dim() == 1:
        dis = dis.unsqueeze(0)
    ranked_dis = torch.argsort(dis, dim=-1)
    closest = ranked_dis[:, 0]
    second_closest = ranked_dis[:, 1]
    first_dis = dis.gather(1, closest.unsqueeze(1)).squeeze(1)
    sec_dis = dis.gather(1, second_closest.unsqueeze(1)).squeeze(1)
    return sec_dis - first_dis, closest.clone().detach()


def get_cluster_id(text, cluster_centers, embedder):
    embedding = embedder.encode(text, convert_to_tensor=True)
    embedding = embedding.reshape(1, -1)
    cluster_id = kmeans_predict(
        embedding,
        cluster_centers=cluster_centers,
        distance='cosine',
        device=device
    )
    return cluster_id


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    data1, data2 = data1.to(device), data2.to(device)
    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)
    cosine = A_normalized * B_normalized
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


def _run_kmeans_once(embeds, k_dim, max_iter, seed):
    dev = embeds.device
    n, d = embeds.shape
    g = torch.Generator(device=dev).manual_seed(seed)
    centers = embeds[torch.randperm(n, generator=g, device=dev)[:k_dim]].clone()
    for _ in range(max_iter):
        emb_n = F.normalize(embeds, dim=-1)
        ctr_n = F.normalize(centers, dim=-1)
        assign = (emb_n @ ctr_n.T).argmax(dim=1)
        prev = centers.clone()
        sums = torch.zeros(k_dim, d, device=dev).index_add_(0, assign, embeds)
        counts = torch.zeros(k_dim, device=dev).index_add_(0, assign, torch.ones(n, device=dev))
        centers = sums / counts.clamp(min=1).unsqueeze(1)
        empty = counts == 0
        if bool(empty.any()):
            ridx = torch.randperm(n, generator=g, device=dev)[:int(empty.sum())]
            centers[empty] = embeds[ridx]
        if float((centers - prev).pow(2).sum(dim=1).sqrt().sum() ** 2) < 1e-4:
            break
    emb_n = F.normalize(embeds, dim=-1)
    cluster_ids = (emb_n @ F.normalize(centers, dim=-1).T).argmax(dim=1)
    return cluster_ids, centers


def get_cluster_centers(embeds, k_dim, max_iter=300, seed=1234, restarts=3):
    """Run cosine KMeans and keep the lowest-inertia restart."""
    best_centers = None
    best_inertia = float('inf')
    for r in range(restarts):
        cluster_ids, centers = _run_kmeans_once(embeds, k_dim, max_iter, seed + r)
        emb_n = F.normalize(embeds, dim=-1)
        ctr_n = F.normalize(centers, dim=-1)
        inertia = float((1 - (emb_n * ctr_n[cluster_ids]).sum(dim=1)).sum())
        is_best = inertia < best_inertia
        print(f"  k-means restart {r + 1}/{restarts}: inertia={inertia:.4f}" + (" *" if is_best else ""))
        if is_best:
            best_inertia = inertia
            best_centers = centers.clone()
    emb_n = F.normalize(embeds, dim=-1)
    cluster_ids = (emb_n @ F.normalize(best_centers, dim=-1).T).argmax(dim=1)
    return cluster_ids, best_centers


def load_embeds(embed_path):
    with open(embed_path, 'rb') as f:
        cache = pickle.load(f)
    if not isinstance(cache, dict) or set(cache) != {"text"}:
        raise ValueError(
            "embedding cache must use the current {'text': Tensor} schema"
        )
    embeddings = cache["text"]
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("embedding cache 'text' value must be a torch.Tensor")
    return embeddings.to(device)


def _main():
    """Run CPU-only primitive smoke tests."""
    passage = (
        "Scientists have recently discovered a new species of deep-sea fish. "
        "The creature was found at a depth of three kilometres. "
        "Its bioluminescent patterns have never been seen before."
    )
    prompt = extract_prompt_from_text(passage, len_prompt=32)
    assert isinstance(prompt, str) and len(prompt) > 0, repr(prompt)
    print(f"extract_prompt_from_text (len_prompt=32): {prompt!r}")

    print("primitives smoke ok")


if __name__ == "__main__":
    _main()
