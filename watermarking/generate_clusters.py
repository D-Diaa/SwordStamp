"""Build centers; adapted from original SemStamp's KMeans utilities."""

import argparse

import torch

from config.cli import add_config_args, resolve
from config.paths import segmentation_cache_tag
from watermarking.primitives import embed_gen_list, get_cluster_centers, load_embeds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Output path for .pt file; default: "
            "<data_path>/cc_{segmentation-identity}_k{k}.pt"
        ),
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=3,
        help="Number of k-means restarts; the run with lowest inertia is kept (default: 3).",
    )
    add_config_args(parser)
    args = parser.parse_args()
    return resolve(args), args.output, args.restarts


def main():
    cfg, output_path, restarts = parse_args()
    data_path = cfg.io.data_path
    if not data_path:
        raise ValueError("No data_path set. Pass it positionally or via io.data_path.")
    embed_path = embed_gen_list(
        data_path,
        cfg.watermark.embedder,
        segmentation_type=cfg.segmentation.type,
        segmentation_backend=cfg.segmentation.backend,
        semcut_max_words=cfg.segmentation.semcut_max_words,
        semcut_window=cfg.segmentation.semcut_window,
        semcut_batch_size=cfg.runtime.semcut_batch_size,
    )
    print(f'Embedding generated at {embed_path}')
    print(f"Generating cluster centers (k={cfg.watermark.sp_dim}, restarts={restarts})..")
    _, cluster_centers = get_cluster_centers(
        load_embeds(embed_path), cfg.watermark.sp_dim, restarts=restarts
    )
    if output_path is None:
        seg_tag = segmentation_cache_tag(
            cfg.segmentation.type,
            cfg.segmentation.backend,
            cfg.segmentation.semcut_max_words,
            cfg.segmentation.semcut_window,
        )
        output_path = (
            f'{data_path}/cc_{seg_tag}_k{cfg.watermark.sp_dim}.pt'
        )
    torch.save(cluster_centers, output_path)
    print(f'Cluster centers saved to {output_path}')


if __name__ == '__main__':
    main()
