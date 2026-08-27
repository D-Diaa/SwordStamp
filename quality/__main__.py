"""Evaluate one derived quality directory with the phased batch engine."""

import argparse
import dataclasses

from config.cli import add_config_args, resolve
from quality.batch import run
from quality.io import derive_io


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    return resolve(parser.parse_args())


def main():
    cfg = parse_args()
    if not cfg.io.data_path:
        raise ValueError("No data_path set. Pass it positionally or via io.data_path.")
    dataset_dir, column, reference, skip_per_pair = derive_io(cfg)
    quality = dataclasses.replace(
        cfg.quality,
        column=column,
        reference=reference,
        skip_per_pair=skip_per_pair,
    )
    run([dataset_dir], dataclasses.replace(cfg, quality=quality), force=True)


if __name__ == '__main__':
    main()
