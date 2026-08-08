"""Strain ablation curve, reworked per advisor feedback on
model_comparison_strain_ablation.py's plot:

X-axis: hit depth ("steps") divided by the LOCAL thickness of the part
right before the hit, not the full original rod length -- the rod-length
version understates strain by ~3x since a hit only locally compresses the
rod's cross-section (diameter ~3in), not its full length (~9in). Local
thickness is estimated per-hit as the y-extent (the canonical compression
axis -- see below) of the k mesh points closest to the press's central
axis, i.e. the material directly under the press, in its pre-hit state.
This naturally captures cumulative compression from earlier hits in the
same series (thickness < the pristine 3in diameter for later hits).

(A second y-axis showing MED as a percent of hit depth was tried here too,
but was dropped -- plotting both an absolute and a relative error curve
together on twin axes read as conflating two different results rather than
clarifying either one.)

Why local thickness is easy to get without re-deriving press geometry:
process_data.py's process_series() already transforms every hit's pre-hit
mesh into a canonical frame centered on the press contact point before
storing it (`pv_mesh_t.points = transform_points_np(..., r_tp1, p_tp1)`),
matching the frame `coords_t` (the sampled point cloud fed to the model) is
stored in. In that frame the press's compression direction is always the
canonical y-axis and the contact point is always at local (x=0, z=0) --
confirmed empirically (see local_thickness_in()'s docstring) -- so thickness
falls out of coords_t directly with no extra geometry lookup needed.

This is a new, separate script (not an edit of model_comparison_strain_
ablation.py) so that plot's known-good output stays reproducible as-is.

Run from this directory so the `main`/`forge_net` imports resolve:
    cd models/forge-net/forge_net && python strain_ablation_local_thickness.py
"""
from pathlib import Path

import numpy as np
import pyvista as pv
import yaml
import matplotlib.pyplot as plt

from forge_net.strain_ablation import per_hit_mean_euclidean_distance, gaussian_kernel_smooth
from forge_net.utils.common import get_project_root
from model_comparison_strain_ablation import load_trainer, latest_checkpoint_step

pv.OFF_SCREEN = True  # this box has no X server; PyVista falls back to software rendering

# (run folder name, plot label, color) -- same trained runs as
# model_comparison_strain_ablation.py.
RUNS = [
    ("jax_chamfer_2048_unmasked_unseeded_lines100k_200ep", "Chamfer-trained", "red"),
    ("jax_mse_2048_unmasked_unseeded_lines100k_200ep", "MSE-trained", "mediumpurple"),
]

N_SAMPLE_HITS = None  # None = use every hit in the dataset
INFERENCE_BATCH_SIZE = 512
K_NEAREST_FOR_THICKNESS = 40  # mesh points nearest the press axis used to estimate local thickness

KERNEL_BANDWIDTH_PCT = 0.15  # wider than model_comparison_strain_ablation.py's 0.08 since
                              # local strain spans a wider range (see MAX_STRAIN_PCT below)
N_GRID_POINTS = 300
MAX_STRAIN_PCT = 10.0  # local strain's 99th percentile is ~7.8% (vs. the rod-length metric's
                        # ~4.5% max) -- beyond this, sample counts get too sparse for a
                        # reliable local estimate

OUT_DIR = get_project_root() / "runs" / "model_comparison"


def local_thickness_in(coords_t_batch: np.ndarray, k: int = K_NEAREST_FOR_THICKNESS) -> np.ndarray:
    """Local thickness of the part directly under the press, before the hit.

    process_data.py's process_series() transforms each hit's pre-hit mesh so
    the press's compression direction is always the canonical y-axis and the
    press contact point is always at local (x=0, z=0) -- confirmed empirically
    against a fresh proc_cyl_stock (pristine diameter 3.0in): points near
    local (x=0, z=0) have a y-extent of ~2.99-3.00in for early-series hits,
    narrowing for later hits in the same series as prior compression
    accumulates. So thickness is just the y-extent of the k points closest to
    the local press axis (by x-z radial distance), no extra geometry lookup
    (press width/position/rotation) needed.

    coords_t_batch: (B, P, 3) pre-hit point cloud, same frame as npz["coords_t"].
    """
    r_xz = np.sqrt(coords_t_batch[..., 0] ** 2 + coords_t_batch[..., 2] ** 2)  # (B, P)
    idx = np.argpartition(r_xz, k, axis=1)[:, :k]  # (B, k) nearest-k indices
    y_near = np.take_along_axis(coords_t_batch[..., 1], idx, axis=1)  # (B, k)
    return y_near.max(axis=1) - y_near.min(axis=1)


def local_strain_pct(npz, sample_idx: np.ndarray, steps: np.ndarray) -> np.ndarray:
    coords_t = npz["coords_t"]
    thickness = np.empty(len(sample_idx), dtype=np.float64)
    for start in range(0, len(sample_idx), INFERENCE_BATCH_SIZE):
        batch_idx = sample_idx[start:start + INFERENCE_BATCH_SIZE]
        pts = np.asarray(coords_t[batch_idx], dtype=np.float64)
        thickness[start:start + len(batch_idx)] = local_thickness_in(pts)
    return steps[sample_idx] / thickness * 100.0


def compute_curves(run_name: str, x_local_strain: np.ndarray,
                    sample_idx: np.ndarray, steps: np.ndarray, npz):
    base_path = get_project_root()
    run_folder = base_path / "runs" / run_name
    with open(run_folder / "config_out.yml") as f:
        config = yaml.safe_load(f)
    config["run"]["run_folder"] = str(run_folder)

    checkpoint_step = latest_checkpoint_step(run_folder)
    print(f"  using checkpoint {checkpoint_step}")
    trainer = load_trainer(config, run_folder, checkpoint_step)

    mean_dist = per_hit_mean_euclidean_distance(trainer, npz, sample_idx, steps)

    in_range = x_local_strain <= MAX_STRAIN_PCT
    x_abs, y_abs = x_local_strain[in_range], mean_dist[in_range]
    x_grid = np.linspace(0, MAX_STRAIN_PCT, N_GRID_POINTS)
    abs_mean, abs_std = gaussian_kernel_smooth(x_abs, y_abs, x_grid, KERNEL_BANDWIDTH_PCT)

    return x_grid, abs_mean, abs_std


def plot_comparison(curves: list, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    for x_grid, abs_mean, abs_std, label, color in curves:
        ax.fill_between(
            x_grid, np.maximum(abs_mean - abs_std, 0), abs_mean + abs_std,
            color=color, alpha=0.15, linewidth=0,
        )
        ax.plot(x_grid, abs_mean, color=color, linewidth=2, label=f"{label} (mean ± 1 std)")

    ax.set_xlim(0, MAX_STRAIN_PCT)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Local strain (hit depth / local pre-hit thickness, %)")
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
    base_path = get_project_root()
    with open(base_path / "runs" / RUNS[0][0] / "config_out.yml") as f:
        shared_config = yaml.safe_load(f)
    npz = np.load(shared_config["datasets"]["data_out"], mmap_mode="r")
    steps = np.asarray(npz["steps"]).reshape(-1)

    n = len(steps)
    if N_SAMPLE_HITS is None:
        sample_idx = np.arange(n)
    else:
        rng = np.random.default_rng(0)
        sample_idx = np.sort(rng.choice(n, size=min(N_SAMPLE_HITS, n), replace=False))

    print("Computing local strain (hit depth / local pre-hit thickness)...")
    x_local_strain = local_strain_pct(npz, sample_idx, steps)

    curves = []
    for run_name, label, color in RUNS:
        print(f"\n=== {label} ({run_name}) ===")
        x_grid, abs_mean, abs_std = compute_curves(
            run_name, x_local_strain, sample_idx, steps, npz,
        )
        curves.append((x_grid, abs_mean, abs_std, label, color))

    plot_comparison(curves, OUT_DIR / "strain_ablation_local_thickness.png")


if __name__ == "__main__":
    main()
