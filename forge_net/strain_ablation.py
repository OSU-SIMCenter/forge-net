"""Strain ablation study for the `jax_mse_2048_unmasked_unseeded` ForgeNet
autoencoder.

Part 1 reports the translation and displacement ranges present in this
model's own training dataset, converting displacement to percent strain of
the original (pre-hit) rod length.

Part 2 samples hits from that same dataset, runs them through the trained
model, and plots per-hit reconstruction error as a function of strain
magnitude.

Scope note: this checkpoint's action feature is "steps" (the solver's
accumulated compression displacement, see data/process_data.py), not the
newer forge_common `hits` table's rho/phi/z/duration action space
(data/process_data_forge_common.py) -- the two are physically different
quantities and action_dims don't even match (1 vs 3-4), so forge_common.db's
hits can't be fed through this checkpoint. The dataset used here
(noisy_cogging.db/random_hits.db, pre-processed into
data/datasets/2048_unseeded_unmasked_data100k_tri_ids.npz per config_out.yml)
is this specific model's own training data, and is the only "hits" source
whose action convention actually matches what it was trained on.

Part 3 tests out-of-distribution generalization: fresh, isolated hits at
strain magnitudes well beyond the training data's max (~4.5%), run directly
through jax_forge itself (not a pre-processed dataset) via the same setup
`heuristic_cogging.py` uses to generate this checkpoint's own training data
(see that file and jax_forge/notebooks/cogging_decomp_weighted_sampling.ipynb
for the reference usage this is adapted from). forge_common is intentionally
not involved here either -- see the scope note above.

Run from this directory so the `main`/`forge_net` imports resolve:
    cd models/forge-net/forge_net && python strain_ablation.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import yaml
import matplotlib.pyplot as plt

from forge_net.model.trainer import ForgeNetTrainer
from forge_net.utils.common import get_project_root
from forge_net.utils.math import barycentric_sampling_np, update_barycentric_points_np
from forge_net.utils.plotting import compare_vector_fields

pv.OFF_SCREEN = True  # this box has no X server; PyVista falls back to software rendering

RUN_NAME = "jax_mse_2048_unmasked_unseeded_backup"
CHECKPOINT_STEP = 248
N_SAMPLE_HITS = None  # None = use every hit in the dataset (inference is cheap; no reason to subsample)
INFERENCE_BATCH_SIZE = 512
SEED = 0
KERNEL_BANDWIDTH_PCT = 0.08  # Gaussian-kernel bandwidth, in % strain, for the smoothed mean/std
N_GRID_POINTS = 300
MAX_STRAIN_PCT = 3.5  # beyond this, sample counts get too sparse (<~100 hits) for a
                       # reliable local estimate -- see report_ranges' printed histogram

OOD_STRAIN_PCTS = [1, 5, 10, 12, 15]  # % strain, chosen to span well past the training max (~4.5%)
OOD_N_POINTS = 2048  # match the checkpoint's trained point_size
OOD_SIM_TIMEOUT_S = 280

# The single canonical stock mesh every training series starts from
# (reused unscaled across the whole noisy_cogging/random_hits dataset, see
# jax-forge/jax_forge/cogging/simple_cogging.py's `mesh_dict["proc_cyl_stock"]`)
# -- its x-extent is the rod's original, pre-hit length. Units: this toy-scale
# dataset (and simple_cogging.py's whole simulator) is built in INCHES, not
# the mm used by forge_common/real_scale.py's real-scale stock -- the two are
# unrelated scales/conventions.
STOCK_MESH_PATH = (
    Path(__file__).resolve().parents[3]
    / "jax-forge" / "jax_forge" / "meshes" / "json" / "proc_cyl_stock.json"
)


def original_rod_length_in() -> float:
    with open(STOCK_MESH_PATH) as f:
        verts = np.array(json.load(f)["Vertices"]).reshape(-1, 3)
    return float(verts[:, 0].max() - verts[:, 0].min())


def report_ranges(npz, rod_length_in: float) -> np.ndarray:
    positions = np.asarray(npz["positions"])       # (N, 3) translation, in
    steps = np.asarray(npz["steps"]).reshape(-1)    # (N,) displacement, in
    percent_strain = steps / rod_length_in * 100.0

    print(f"Original rod length (proc_cyl_stock.json x-extent): {rod_length_in:.4f} in")
    print(f"Dataset: {len(steps)} hits from {len(npz['series_lengths'])} series\n")

    print("Translation (press position vector) range per axis, in:")
    for axis, name in enumerate("xyz"):
        col = positions[:, axis]
        print(f"  {name}: [{col.min():.4f}, {col.max():.4f}]")

    print(f"\nDisplacement ('steps', solver-accumulated compression) range: "
          f"[{steps.min():.4f}, {steps.max():.4f}] in")
    print(f"Percent strain of the {rod_length_in:.3f} in original rod: "
          f"[{percent_strain.min():.3f}%, {percent_strain.max():.3f}%]\n")
    return percent_strain


def load_trainer(config: dict, run_folder: Path) -> ForgeNetTrainer:
    base_path = run_folder.parents[1]  # forge_net/ project root, for `import main`
    sys.path.insert(0, str(base_path))
    from main import make_dataloaders  # noqa: E402

    train_loader, _test_loader = make_dataloaders(config)
    trainer = ForgeNetTrainer(config, train_loader, log_to_tb=False)
    trainer.load(model_path=run_folder / "checkpoints" / str(CHECKPOINT_STEP))
    return trainer


def per_hit_mean_euclidean_distance(
    trainer: ForgeNetTrainer, npz, sample_idx: np.ndarray, steps: np.ndarray,
) -> np.ndarray:
    """Per-hit mean Euclidean distance between predicted and ground-truth
    next-state point clouds -- the same metric `eval.py`'s `evaluate_series`
    already reports (`np.linalg.norm(x_tp1_hat - x_tp1_gt, axis=-1)`, then
    averaged over points), in physical inches rather than MSE's squared in^2.
    """
    coords_t = npz["coords_t"]
    coords_tp1 = npz["coords_tp1"]

    mean_dist = np.empty(len(sample_idx), dtype=np.float64)
    for start in range(0, len(sample_idx), INFERENCE_BATCH_SIZE):
        batch_idx = sample_idx[start:start + INFERENCE_BATCH_SIZE]
        x_t = np.asarray(coords_t[batch_idx], dtype=np.float32)
        x_tp1 = np.asarray(coords_tp1[batch_idx], dtype=np.float32)
        a = steps[batch_idx].reshape(-1, 1).astype(np.float32)

        delta_gt = x_tp1 - x_t
        delta_hat = np.asarray(trainer.predict(x_t, a)) / 100.0
        point_dist = np.linalg.norm(delta_hat - delta_gt, axis=-1)  # (B, 2048)
        mean_dist[start:start + len(batch_idx)] = point_dist.mean(axis=-1)
    return mean_dist


def gaussian_kernel_smooth(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, bandwidth: float):
    """Nadaraya-Watson kernel regression: a continuous local weighted mean/std
    of `y` as a function of `x`, evaluated at `x_grid`. Every sample
    contributes to every grid point with a smoothly decaying Gaussian
    weight, so unlike binning there are no hard edges or discrete jumps --
    the estimate is a genuinely continuous function of x. Where data is
    sparse (the long strain tail here), the effective weighted sample size
    drops and the estimate is noisier by construction, rather than being
    truncated or forced into one wide bin.
    """
    diffs = (x_grid[:, None] - x[None, :]) / bandwidth  # (n_grid, n_samples)
    weights = np.exp(-0.5 * diffs ** 2)
    weight_sum = weights.sum(axis=1)
    mean = (weights @ y) / weight_sum
    var = (weights * (y[None, :] - mean[:, None]) ** 2).sum(axis=1) / weight_sum
    return mean, np.sqrt(var)


def plot_error_vs_strain(percent_strain: np.ndarray, mean_dist: np.ndarray, out_path: Path):
    in_range = percent_strain <= MAX_STRAIN_PCT
    x, y = percent_strain[in_range], mean_dist[in_range]

    # No raw scatter here -- the dataset's cogging policy was run many times
    # at similar depths, so a per-point scatter is dominated by an artifact
    # of how the data was generated (a dense pile-up around ~0.5% strain),
    # not by anything about the model. The shaded std band already conveys
    # spread, so it's the more honest single signal to show.
    x_grid = np.linspace(x.min(), x.max(), N_GRID_POINTS)
    mean_curve, std_curve = gaussian_kernel_smooth(x, y, x_grid, KERNEL_BANDWIDTH_PCT)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.fill_between(
        x_grid, np.maximum(mean_curve - std_curve, 0), mean_curve + std_curve,
        color="#1B2A4A", alpha=0.15, linewidth=0, label="local mean ± 1 std",
    )
    ax.plot(x_grid, mean_curve, color="#1B2A4A", linewidth=2, label="local mean")

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


def run_fresh_jax_forge_hit(strain_pct: float, rod_length_in: float, seed: int = SEED,
                             timeout: float = OOD_SIM_TIMEOUT_S):
    """Runs ONE isolated hit on a fresh proc_cyl_stock at a target depth
    corresponding to `strain_pct`, entirely through jax_forge (mirroring
    `heuristic_cogging.py`'s/the weighted-sampling notebook's per-hit call --
    forge_common is not involved, see the module docstring). The pre/post
    point clouds are built the same way `process_data.py` builds ForgeNet's
    training pairs (barycentric sampling + triangle-id correspondence), so
    the model sees data shaped exactly like what it was trained on -- only
    the strain magnitude is out of distribution.

    Translation is fixed at the origin (press centered on the fresh stock)
    and rotation is identity, so strain magnitude is the only varying
    factor -- position/orientation effects aren't part of this test.

    Returns (coords_t, coords_tp1, actual_depth), or None if jax_forge's
    adaptive solver failed outright at this depth (see sim_handler.py's
    `[-1]` failure sentinel) -- large depths can legitimately fail; that's a
    valid experimental outcome, not a bug to hide. `actual_depth` is the sum
    of the solver's own substeps, which can fall short of the requested
    depth if it runs out of the time budget (`timeout`) before finishing --
    the caller compares it against the requested depth to detect that.
    """
    from jax_forge.lib.mesh_container import MeshContainer
    from jax_forge.lib.press_info import PressInfo
    from jax_forge.lib.sim_handler import SimulationHandler
    from jax_forge.utils.utils import eulerxyz_to_quat

    # press_id=2 ("XL Hammer Y") from jax_forge's own DBMS default seed data
    # (data_storage/database_handler.py's `default_presses`) -- hardcoded
    # here since no local database.db exists in this checkout; these are
    # fixed seed values, not anything fitted/learned.
    press_info = PressInfo("XL Hammer Y", 1.0, 3.0, [0, 1, 0])
    identity_quat = list(eulerxyz_to_quat([0, 0, 0]))
    depth = strain_pct / 100.0 * rod_length_in

    proc_mesh = MeshContainer.from_json(STOCK_MESH_PATH)
    # `proc_mesh.obj` right after from_json() is the raw JSON surface, NOT
    # the topology the solver will actually use: execute_simulation_
    # adaptive_step() calls self.mesh.get_surface() after every step, which
    # re-derives the surface from the volumetric (tetra) mesh's boundary
    # faces -- a different triangulation (different triangle count/
    # numbering) than the raw JSON. Force the pre-hit surface onto that same
    # boundary-derived topology now, so triangle IDs sampled here stay valid
    # for the post-hit surface (same tetra connectivity => same boundary
    # topology throughout a single hit, since there's no mid-hit remeshing).
    proc_mesh.get_surface()
    pv_mesh_t = pv.from_meshio(proc_mesh.obj).extract_surface()
    # `sim_handler` mutates `proc_mesh` (hence `proc_mesh.obj`) in place as
    # it steps, so copy the pre-hit surface out before running anything.
    pv_mesh_t = pv.PolyData(np.array(pv_mesh_t.points, dtype=np.float64).copy(),
                             np.array(pv_mesh_t.faces).copy())
    coords_t, tri_ids, bary_coords = barycentric_sampling_np(pv_mesh_t, OOD_N_POINTS, seed=seed)

    sim_handler = SimulationHandler(press_info=press_info, mesh_data=proc_mesh)
    out_mesh = sim_handler.execute_simulation_adaptive_step(
        force=float(depth), translation_vector=[0, 0, 0], quaternion=identity_quat, timeout=timeout,
    )
    if isinstance(out_mesh, dict):  # hard-failure sentinel (sim_handler.py)
        return None

    unity_result = sim_handler.get_unity_result()
    actual_depth = float(np.sum(unity_result["Steps"]))

    pv_mesh_tp1 = pv.from_meshio(out_mesh.obj).extract_surface()
    coords_tp1 = update_barycentric_points_np(pv_mesh_tp1, tri_ids, bary_coords)
    return coords_t.astype(np.float32), coords_tp1.astype(np.float32), actual_depth


def run_ood_generalization_experiment(trainer: ForgeNetTrainer, rod_length_in: float, out_dir: Path):
    print("\n=== Part 3: out-of-distribution generalization (fresh jax_forge hits) ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    for pct in OOD_STRAIN_PCTS:
        target_depth = pct / 100.0 * rod_length_in
        print(f"\nStrain {pct}% -> target depth {target_depth:.4f} in")
        result = run_fresh_jax_forge_hit(pct, rod_length_in)
        if result is None:
            print(f"  jax_forge solver failed to converge at {pct}% strain -- skipping")
            continue
        coords_t, coords_tp1_gt, actual_depth = result
        if actual_depth < target_depth - 1e-4:
            actual_pct = actual_depth / rod_length_in * 100.0
            print(f"  solver only reached {actual_depth:.4f} in ({actual_pct:.2f}% strain, "
                  f"{actual_depth / target_depth:.1%} of target) within the {OOD_SIM_TIMEOUT_S}s "
                  f"budget -- reporting the ACTUAL achieved strain, not the requested one")

        x_t = coords_t[None]
        a = np.array([[actual_depth]], dtype=np.float32)
        delta_hat = np.asarray(trainer.predict(x_t, a)) / 100.0
        coords_tp1_hat = coords_t + delta_hat[0]

        mean_dist = np.linalg.norm(coords_tp1_hat - coords_tp1_gt, axis=-1).mean()
        print(f"  jax_forge vs. ForgeNet mean Euclidean distance: {mean_dist:.4f} in")

        fig_path = out_dir / f"ood_strain_{pct}pct_vector_field.png"
        compare_vector_fields(coords_t, coords_tp1_gt, coords_tp1_hat, fig_path=str(fig_path))
        print(f"  Saved vector field comparison -> {fig_path}")


def main():
    base_path = get_project_root()
    run_folder = base_path / "runs" / RUN_NAME
    config_path = run_folder / "config_out.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["run"]["run_folder"] = str(run_folder)  # config_out.yml's own value is stale

    rod_length_in = original_rod_length_in()
    npz = np.load(config["datasets"]["data_out"], mmap_mode="r")
    steps = np.asarray(npz["steps"]).reshape(-1)

    percent_strain_all = report_ranges(npz, rod_length_in)

    n = len(percent_strain_all)
    if N_SAMPLE_HITS is None:
        sample_idx = np.arange(n)
    else:
        rng = np.random.default_rng(SEED)
        sample_idx = np.sort(rng.choice(n, size=min(N_SAMPLE_HITS, n), replace=False))

    trainer = load_trainer(config, run_folder)
    mean_dist = per_hit_mean_euclidean_distance(trainer, npz, sample_idx, steps)

    plot_error_vs_strain(
        percent_strain_all[sample_idx], mean_dist,
        run_folder / "eval" / "strain_ablation.png",
    )

    run_ood_generalization_experiment(trainer, rod_length_in, run_folder / "eval" / "ood_generalization")


if __name__ == "__main__":
    main()
