"""Derive stage-dependent quality inputs without importing the ML stack."""
from config.paths import target_dir, watermark_dir


def derive_io(cfg):
    """Return the target directory, columns, reference, and pairwise policy."""
    target = cfg.io.target
    dataset_dir = cfg.io.output_dir or target_dir(cfg)
    column = cfg.quality.column or ("text" if target == "watermark" else "para_text")
    if cfg.quality.reference is not None:
        reference = cfg.quality.reference
    else:
        reference = cfg.io.data_path if target == "watermark" else watermark_dir(cfg)
    skip_per_pair = cfg.quality.skip_per_pair
    if skip_per_pair is None:
        skip_per_pair = target == "watermark"
    return dataset_dir, column, reference, skip_per_pair
