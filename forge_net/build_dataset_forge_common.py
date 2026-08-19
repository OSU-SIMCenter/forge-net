"""Standalone CPU-only driver for `process_data_forge_common.make_dataset`'s
parallel `extract_data` path -- a SEPARATE process from `train_forge_common.
py`, specifically so the `multiprocessing.Pool` workers it spawns never
touch the real GPU `train_forge_common.py` needs for training afterward.

Why this can't just be `train_forge_common.py --workers N`: `forge_net.
utils.math` (imported by `process_data_forge_common`) imports `jax`, and
Python's `multiprocessing` `spawn` start method unconditionally re-imports
the launching `__main__` module (and therefore this whole import chain,
INCLUDING jax) in every worker BEFORE a `Pool(initializer=...)` callback
ever runs -- there is no reliable way to force those workers onto CPU-only
from inside a process whose own `__main__` already needs the GPU. Setting
`CUDA_VISIBLE_DEVICES` as the literal first line of THIS script's `__main__`
instead means every spawned worker (which re-imports THIS script) sees it
too, before jax ever initializes -- same pattern as `forge_common/scripts/
run_cogging_dataset_thermal.py`.

`make_dataset` skips regeneration if `datasets.data_out` already exists, so
running this first and then `train_forge_common.py` normally is a strict
speedup, not a different code path -- the training process's own inline
`make_dataset` call just finds the `.npz` already there.

Usage (from `models/forge-net`):
    python -m forge_net.build_dataset_forge_common --config forge_net/configs/<name>.yml --workers 32
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU -- see module docstring; must happen before any jax import

import argparse

import yaml

from forge_net.data.process_data_forge_common import make_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    parser.add_argument("--workers", type=int, default=32, help="parallel worker PROCESSES (CPU-only, see module docstring)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    make_dataset(config, n_workers=args.workers)


if __name__ == "__main__":
    main()
