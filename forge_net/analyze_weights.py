"""Weight-magnitude distribution analysis for a trained ForgeNet checkpoint
-- how many parameters exceed common magnitude thresholds, plus histograms
saved into the run's own folder.

Usage (from `models/forge-net`):
    python -m forge_net.analyze_weights --config forge_net/configs/forge_common_thermal_v2_60k_canon.yml
"""

import argparse
from pathlib import Path

import jax
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from forge_net.eval_thermal import _load_trained_trainer  # noqa: E402
from forge_net.utils.common import get_project_root  # noqa: E402

_THRESHOLDS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def analyze_weights(config: dict, trainer) -> dict:
    """`trainer.state.params` is the SAME pytree `Total trainable parameters`
    counting already flattens via `jax.tree_util.tree_leaves` (trainer.py's
    `_make_network`) -- reused here, not a separate traversal, so the total
    count matches what training itself printed."""
    flat_params = jax.tree_util.tree_leaves(trainer.state.params)
    all_weights = np.concatenate([np.asarray(p).ravel() for p in flat_params])
    abs_weights = np.abs(all_weights)
    total = len(all_weights)

    print(f"Total parameters: {total:,}")
    counts = {}
    for t in _THRESHOLDS:
        n = int(np.sum(abs_weights > t))
        counts[t] = n
        print(f"  |w| > {t:g}: {n:,} ({100 * n / total:.2f}%)")

    out_dir = Path(config["run"]["run_folder"])
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Raw signed value distribution -- shape/spread of the trained weights.
    axes[0].hist(all_weights, bins=200, color="steelblue")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("weight value")
    axes[0].set_ylabel("count (log scale)")
    axes[0].set_title("Raw weight distribution")

    # Log-magnitude distribution -- directly shows where mass sits relative
    # to the |w| > threshold cutoffs printed above (dashed reference lines).
    nonzero = abs_weights[abs_weights > 0]
    axes[1].hist(np.log10(nonzero), bins=200, color="darkorange")
    ymax = axes[1].get_ylim()[1]
    for t in _THRESHOLDS:
        axes[1].axvline(np.log10(t), color="gray", linestyle="--", alpha=0.6)
        axes[1].text(np.log10(t), ymax * 0.97, f"{t:g}", rotation=90, fontsize=7, va="top", ha="right")
    axes[1].set_xlabel("log10(|weight|)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Weight magnitude distribution (log10)")

    fig.suptitle(f"Weight distribution -- {config['run']['run_name']} ({total:,} total params)")
    fig.tight_layout()
    fig_path = out_dir / "weight_distribution.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Saved -> {fig_path}")

    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if not Path(config["run"]["run_folder"]).is_absolute():
        config["run"]["run_folder"] = str(get_project_root() / "runs" / config["run"]["run_name"])

    trainer, data = _load_trained_trainer(config)
    analyze_weights(config, trainer)


if __name__ == "__main__":
    main()
