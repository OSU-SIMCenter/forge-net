"""Strain ablation curve overlaying jax_chamfer_2048 vs. jax_mse_2048 predictions.

Both runs were trained on the same dataset (2048_unseeded_unmasked_data100k_
tri_ids.npz), so their per-hit reconstruction error can be computed against
the same held-out strain values and plotted on shared axes for a direct
comparison. See strain_ablation.py's module docstring for the full scope
notes on what this dataset/action space does and doesn't cover -- this script
reuses that file's data-loading and metric logic verbatim.

Run from this directory so the `main`/`forge_net` imports resolve:
    cd models/forge-net/forge_net && python model_comparison_strain_ablation.py
"""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import yaml
import matplotlib.pyplot as plt

from forge_net.model.trainer import ForgeNetTrainer
from forge_net.strain_ablation import (
    original_rod_length_in,
    per_hit_mean_euclidean_distance,
    gaussian_kernel_smooth,
)
from forge_net.utils.common import get_project_root

pv.OFF_SCREEN = True  # this box has no X server; PyVista falls back to software rendering

# (run folder name, plot label, color). Checkpoint step is auto-detected as
# the highest epoch number under runs/<name>/checkpoints/ -- trainer.py only
# ever checkpoints on a test-loss improvement, so the highest-numbered
# checkpoint is always the best/latest one.
RUNS = [
    ("jax_chamfer_2048_unmasked_unseeded_lines100k_200ep", "Chamfer-trained", "red"),
    ("jax_mse_2048_unmasked_unseeded_lines100k_200ep", "MSE-trained", "mediumpurple"),
]

N_SAMPLE_HITS = None  # None = use every hit in the dataset
INFERENCE_BATCH_SIZE = 512
SEED = 0
KERNEL_BANDWIDTH_PCT = 0.08  # Gaussian-kernel bandwidth, in % strain, for the smoothed mean/std
N_GRID_POINTS = 300
MAX_STRAIN_PCT = 5.0  # x-axis extent; the dataset's own max is ~4.5%, so the
                       # curve/band naturally end before the axis does

OUT_DIR = get_project_root() / "runs" / "model_comparison"


def latest_checkpoint_step(run_folder: Path) -> int:
    ckpt_dir = run_folder / "checkpoints"
    steps = [int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return max(steps)


def load_trainer(config: dict, run_folder: Path, checkpoint_step: int) -> ForgeNetTrainer:
    base_path = run_folder.parents[1]  # forge_net/ project root, for `import main`
    sys.path.insert(0, str(base_path))
    from main import make_dataloaders  # noqa: E402

    train_loader, _test_loader = make_dataloaders(config)
    trainer = ForgeNetTrainer(config, train_loader, log_to_tb=False)
    trainer.load(model_path=run_folder / "checkpoints" / str(checkpoint_step))
    return trainer


def compute_curve(run_name: str, percent_strain_all: np.ndarray,
                   sample_idx: np.ndarray, steps: np.ndarray, npz):
    base_path = get_project_root()
    run_folder = base_path / "runs" / run_name
    with open(run_folder / "config_out.yml") as f:
        config = yaml.safe_load(f)
    config["run"]["run_folder"] = str(run_folder)  # config_out.yml's own value can be stale

    checkpoint_step = latest_checkpoint_step(run_folder)
    print(f"  using checkpoint {checkpoint_step}")
    trainer = load_trainer(config, run_folder, checkpoint_step)

    mean_dist = per_hit_mean_euclidean_distance(trainer, npz, sample_idx, steps)

    x = percent_strain_all[sample_idx]
    in_range = x <= MAX_STRAIN_PCT
    x, y = x[in_range], mean_dist[in_range]
    x_grid = np.linspace(x.min(), x.max(), N_GRID_POINTS)
    mean_curve, std_curve = gaussian_kernel_smooth(x, y, x_grid, KERNEL_BANDWIDTH_PCT)
    return x_grid, mean_curve, std_curve


def plot_comparison(curves: list, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    for x_grid, mean_curve, std_curve, label, color in curves:
        ax.fill_between(
            x_grid, np.maximum(mean_curve - std_curve, 0), mean_curve + std_curve,
            color=color, alpha=0.15, linewidth=0,
        )
        ax.plot(x_grid, mean_curve, color=color, linewidth=2, label=f"{label} (mean ± 1 std)")

    ax.set_xlim(0, MAX_STRAIN_PCT)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Strain magnitude (% of original rod length)")
    ax.set_ylabel("Reconstruction MED (in)")
    ax.grid(True, alpha=0.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved plot -> {out_path}")


def main():
    # Both runs share the same dataset, so load it and derive strain once.
    base_path = get_project_root()
    with open(base_path / "runs" / RUNS[0][0] / "config_out.yml") as f:
        shared_config = yaml.safe_load(f)
    rod_length_in = original_rod_length_in()
    npz = np.load(shared_config["datasets"]["data_out"], mmap_mode="r")
    steps = np.asarray(npz["steps"]).reshape(-1)
    percent_strain_all = steps / rod_length_in * 100.0

    n = len(percent_strain_all)
    if N_SAMPLE_HITS is None:
        sample_idx = np.arange(n)
    else:
        rng = np.random.default_rng(SEED)
        sample_idx = np.sort(rng.choice(n, size=min(N_SAMPLE_HITS, n), replace=False))

    curves = []
    for run_name, label, color in RUNS:
        print(f"\n=== {label} ({run_name}) ===")
        x_grid, mean_curve, std_curve = compute_curve(
            run_name, percent_strain_all, sample_idx, steps, npz,
        )
        curves.append((x_grid, mean_curve, std_curve, label, color))

    plot_comparison(curves, OUT_DIR / "strain_ablation_comparison.png")


if __name__ == "__main__":
    main()
