"""Semantic entropy adapted from original SemStamp's ``eval_quality.py``."""

import os
import pickle
from collections import Counter

import numpy as np


def evaluate_semantic_entropy(model, gens, tokenizer, ref_texts, qcfg, save_testgen_path):
    import faiss
    import torch
    from tqdm import trange

    # Cache generation embeddings under save_testgen_path.
    load_kmeans_path = qcfg.load_kmeans_path
    save_kmeans_path = qcfg.corpus
    load_testgen_path = qcfg.load_testgen_path
    mode = qcfg.sem_ent_mode
    cluster_size = qcfg.cluster_size

    # Step 1: Perform clustering (if no centroids provided)
    def batch_tokenize(texts):
        input_ids = tokenizer(
            list(texts),
            return_tensors="pt",
            padding="longest",
        ).input_ids.to("cuda")
        return input_ids.unsqueeze(1)  # shape: num of text X longest X embed dim

    @torch.no_grad()
    def embed_gen(texts, desc="Featurizing", mode="last_token", cluster_size=500):
        # Hidden-state features need no autograd graph.
        input_ids = batch_tokenize(texts)
        sent_len = [
            len(tokenizer(text, return_tensors="pt").input_ids[0])
            for text in texts
        ]
        out_embeds = []
        if mode == "last_token":
            out_embeds = [
                model(input_ids[i], output_hidden_states=True).hidden_states[-1][0][sent_len[i] - 1]
                .cpu()
                .detach()
                .squeeze()
                for i in trange(len(input_ids), desc=desc)
            ]
        elif mode == "last_mean_pooling":
            for i in trange(len(input_ids), desc=desc):
                out_embeds.append(
                    torch.mean(
                        [
                            model(input_ids[i], output_hidden_states=True).hidden_states[-1][0][j]
                            .cpu()
                            .detach()
                            .squeeze()
                        ]
                    )
                    for j in range(sent_len)
                )
        elif mode == "all_mean_pooling":
            for i in trange(len(input_ids), desc=desc):
                hidden_states = model(input_ids[i], output_hidden_states=True).hidden_states
                hidden_states_len = len(hidden_states)  # 33
                hidden_states_list = [
                    hidden_states[k][0][sent_len[i] - 1].cpu().detach().squeeze()
                    for k in range(hidden_states_len)
                ]
                # shape=[2560]
                pooled_states = torch.mean(torch.stack(hidden_states_list), dim=0)
                out_embeds.append(pooled_states)
        # out_embeds shape: len(input_ids) X 2560 (hidden_state_size/num of features)
        input_tensor = torch.stack(out_embeds).float().numpy()
        return input_tensor

    # build from scratch if no stored index and centroids

    if load_kmeans_path == None:
        ncentroids = cluster_size
        niter = 100
        verbose = True
        input_tensor = embed_gen(ref_texts, desc="Featurizing train generations", mode=mode)

        d = input_tensor.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        kmeans.train(input_tensor)
        centroids = kmeans.centroids
        index = kmeans.index
        # save
        cpath = os.path.join(save_kmeans_path, f"mode={mode}-cluster_size={cluster_size}-centroids.npy")
        ipath = os.path.join(save_kmeans_path, f"mode={mode}-cluster_size={cluster_size}-index.pkl")
        with open(cpath, "wb") as f:
            np.save(f, centroids)
            f.close()
        with open(ipath, "wb") as f:
            pickle.dump(index, f)
            f.close()
    # Otherwise, load saved index and centroids
    else:
        cpath = os.path.join(load_kmeans_path, f"mode={mode}-cluster_size={cluster_size}-centroids.npy")
        ipath = os.path.join(load_kmeans_path, f"mode={mode}-cluster_size={cluster_size}-index.pkl")
        centroids = np.load(cpath)
        with open(ipath, "rb") as f:
            index = pickle.load(f)
            f.close()

    # Step 2: Generate and assign clusters
    if load_testgen_path == None:
        gen_embed = embed_gen(gens, "Featurizing test generations", mode=mode)
        # save
        gen_embed_path = os.path.join(save_testgen_path, "test-gen-embed.pkl")
        with open(gen_embed_path, "wb") as f:
            pickle.dump(gen_embed, f)
            f.close()
    else:
        gen_embed_path = os.path.join(load_testgen_path, "test-gen-embed.pkl")
        with open(gen_embed_path, "rb") as f:
            gen_embed = pickle.load(f)
            f.close()
    # find 1-closest cluster
    D, I = index.search(gen_embed, 1)

    # Step 3: Calculate entropy
    freq_dict = Counter(list(I.squeeze()))
    freqs = list(freq_dict.values())
    total_freqs = np.sum(freqs)
    approx_p = [f / total_freqs for f in freqs]
    sem_ent = np.sum([-1 * p * np.log(p) for p in approx_p])
    with open(os.path.join(save_testgen_path, f"mode={mode}-cluster_size={cluster_size}-distribution.txt"), "wb") as f:
        print(f"Semantic Ent: {sem_ent}")
        f.close()
    return sem_ent
