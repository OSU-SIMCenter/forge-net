"""Evaluation/animation for a `predict_temperature=True` ForgeNet run --
`eval.py`'s `evaluate`/`evaluate_series` don't know about the temperature
head at all and hard-crash on one (`trainer.predict(...) / 100.0` against a
`(delta_xyz, delta_temp)` tuple, confirmed directly). This module has two
pieces:

`render_series_thermal_gif` -- one series, RECURSIVE rollout (the model's
own predictions feed forward each step, action-conditioned on the real
recorded `rho`/`phi`/`z` sequence), laid out as:

    +----------------+----------------+------------------+
    | GT   -- mesh    | GT   -- points |                  |
    +----------------+----------------+  error over time  |
    | Pred -- mesh    | Pred -- points |  (geom + temp)   |
    +----------------+----------------+------------------+

Left column ("mesh view"): GROUND TRUTH uses the REAL recorded mesh
(`forge_common.data_storage.mesh_snapshot_from_hit_row`, queried directly
from the DB by series_id/step_number -- exact geometry+connectivity, not a
reconstruction). PREDICTION has no real connectivity (ForgeNet predicts an
unconnected point cloud) -- reconstructed via the LSQR method
(`forge_net.invert_deltas.invert_deltas_to_tet_mesh`/`invert_scalar_to_tet_
mesh`): solves a sparse least-squares system for the FULL mesh's (real,
same-as-GT) vertex positions/temperatures that best explain the model's
predicted SAMPLED points, given their known barycentric weights within that
fixed connectivity (see `invert_deltas.py`'s module-level docstring
additions for the exact math -- same approach `invert_deltas_to_mesh`
already used for triangle surfaces, generalized here to `cw_slab_model`'s
tetrahedral volume mesh). `alpha=0.05` graph-Laplacian regularization,
warm-started from the previous step's solve each frame (stabler + faster
than a cold LSQR solve every frame).

Right column ("point view"): the raw sampled point cloud (what the model
actually operates on), colored by temperature.

Error panel (rightmost, full height): mean point-wise position error (mm)
and mean absolute temperature error (degrees C) over the WHOLE rollout on
twin y-axes, with a moving vertical line marking the current frame.

`evaluate_thermal` -- `eval.py`'s `evaluate()` (one-step, non-recursive,
per-`eval_idxs` scatter/vector-field plots), adapted for a 4-channel
(xyz+temp) input and tuple output, REUSING the existing plotting functions
from `forge_net.utils.plotting` unchanged (they already accept an arbitrary
per-point `loss_cont` color signal, see `compare_scatters_w_loss_cont`/
`visualize_vector_diff_w_loss_cont`'s existing calls in `eval.py`) -- run
ONCE with `loss_cont` = squared position error (identical to `evaluate()`),
and a SECOND time with `loss_cont` = squared temperature error, so both
error signals get the same plot treatment.

Usage (from `models/forge-net`):
    python -m forge_net.eval_thermal --config forge_net/configs/forge_common_thermal_v1.yml --series-idx 0 --evaluate
"""

import argparse
import json
from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pyvista as pv
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tqdm import tqdm  # noqa: E402

from forge_net.data.process_data_forge_common import _phi_to_canonical_quat, make_dataloaders, make_dataset  # noqa: E402
from forge_net.invert_deltas import invert_deltas_to_tet_mesh, invert_scalar_to_tet_mesh  # noqa: E402
from forge_net.model.trainer import ForgeNetTrainer  # noqa: E402
from forge_net.utils.common import actions_from_feature_map, get_project_root  # noqa: E402
from forge_net.utils.math import transform_points, untransform_points, untransform_points_np  # noqa: E402
from forge_net.utils.plotting import (  # noqa: E402
    compare_scatters,
    compare_scatters_w_loss_cont,
    compare_vector_fields,
    plot_eval_series,
    plot_spherical_heatmap,
    visualize_point_diff,
    visualize_vector_diff_w_loss_cont,
    visualize_vector_diff_w_scale,
)

from forge_common.data_storage import ForgeDBMS, mesh_snapshot_from_hit_row  # noqa: E402

_CLIM = (20.0, 1000.0)  # same fixed thermal range used throughout forge_common's own renders


def _load_trained_trainer(config: dict) -> "tuple[ForgeNetTrainer, dict]":
    """Rebuilds the SAME dataloaders `train_forge_common.py` used (from the
    already-processed `.npz`, `make_dataset` is a no-op if it exists) and
    restores the LATEST checkpoint for this run -- no training happens
    here. Returns `(trainer, npz_data)`."""
    make_dataset(config=config)
    train_loader, test_loader, loss_norm_stats = make_dataloaders(config)
    if config["network"].get("predict_temperature", False):
        config["network"].setdefault("pos_delta_std", loss_norm_stats["pos_delta_std"])
        config["network"].setdefault("temp_delta_std", loss_norm_stats["temp_delta_std"])
    trainer = ForgeNetTrainer(config, train_loader, test_loader, resume_epoch=1, log_to_tb=False)
    data = np.load(config["datasets"]["data_out"], allow_pickle=True)  # series_ids is dtype=object
    return trainer, data


def _pick_series(data: dict, series_idx: int) -> "tuple[int, int]":
    """`(start, length)` pair-index range for the `series_idx`-th series in
    the flat concatenated npz arrays (see `process_data_forge_common.
    extract_data`'s `series_lengths` convention)."""
    lengths = data["series_lengths"]
    if not (0 <= series_idx < len(lengths)):
        raise ValueError(f"series_idx={series_idx} out of range (0..{len(lengths) - 1})")
    start = int(np.sum(lengths[:series_idx]))
    length = int(lengths[series_idx])
    return start, length


def _load_real_series_meshes(db_path: str, series_id: str):
    """Real recorded meshes for every step of `series_id`, straight from
    the DB -- `(vertices_per_step, temps_per_step, tetra)`. `tetra` is
    returned ONCE (cw_slab_model never remeshes -- fixed connectivity for a
    whole series, confirmed repeatedly elsewhere in this project)."""
    db = ForgeDBMS(db_path)
    rows = db.get_series_hits(series_id)
    db.close()

    vertices_per_step = []
    temps_per_step = []
    tetra = None
    for row in rows:
        mesh = mesh_snapshot_from_hit_row(row)
        vertices_per_step.append(mesh.vertices)
        temps_per_step.append(np.array(json.loads(row["input_temperature"])))
        if tetra is None:
            tetra = mesh.tetra
    return vertices_per_step, temps_per_step, tetra


def _rollout_series(
    trainer: ForgeNetTrainer, coords: np.ndarray, temps: np.ndarray, actions: np.ndarray,
    quats: np.ndarray, positions: np.ndarray,
):
    """RECURSIVE rollout: step 0's input is the real ground-truth `(coords[0],
    temps[0])`; every later step's input is the MODEL'S OWN previous
    prediction, action-conditioned on the real recorded action sequence.

    Per-hit frame canonicalization (`process_data_forge_common.py`'s
    `_phi_to_canonical_quat`): pair `i`'s `coords_t`/`coords_tp1` are BOTH
    rotated by PAIR i's OWN quaternion, so `coords[i]`/`coords[i+1]` (this
    function's `t`/`t+1` inputs) share a frame with EACH OTHER but NOT with
    `coords[i-1]`/`coords[i+2]` etc. -- a recursive rollout must therefore
    re-canonicalize the running state into EACH step's own frame right
    before calling the model, exactly the `untransform_points`/
    `transform_points` frame-hop `eval.py`'s `evaluate_series` already uses
    for the legacy schema's per-hit local frames (`forge_net.utils.math`,
    reused here unchanged rather than duplicated).

    Args:
        coords: (n_steps+1, N, 3) ground-truth positions (`coords_t` of pair
            0, then every pair's `coords_tp1`) -- `coords[0]` in PAIR 0's own
            canonical frame (matches how the model was trained to see it),
            everything else consumed only to seed the rollout.
        temps: (n_steps+1, N) ground-truth temperatures, same indexing
            (unaffected by rotation -- scalars).
        actions: (n_steps, action_dims) -- the action that produced step
            `i+1` from step `i` (`phi` already excluded, see `config
            ["network"]["action_features"]`).
        quats: (n_steps, 4) -- pair `i`'s own WORLD -> CANONICAL quaternion
            (x,y,z,w), i.e. `_phi_to_canonical_quat(phi[i])`.
        positions: (n_steps, 3) -- pair `i`'s own WORLD -> CANONICAL
            translation (zero unless `center_hit_z` was on when the dataset
            was built -- see `process_data_forge_common.py`'s `canon_pos`).
            Reading the REAL stored value here (not a hardcoded zero) is
            what makes this correct for a `center_hit_z=True` dataset, not
            just the rotation-only (`center_hit_z=False`) ones.

    Returns `(pred_coords, pred_temps)`, both `(n_steps+1, ...)`, ALL WORLD
    FRAME (unlike `coords`/`temps` above) -- needed downstream for LSQR
    reconstruction against the real world-frame base mesh and so the
    mesh-view panel lines up with the real (world-frame, straight from the
    DB) ground-truth mesh. Index 0 equals the real ground-truth start
    (world-framed via `untransform_points_np`, nothing to predict there)."""
    n_steps = actions.shape[0]

    pred_coords = [untransform_points_np(coords[0], quats[0], positions[0])]
    pred_temps = [temps[0]]

    x_t = jnp.concatenate([jnp.asarray(coords[0]), jnp.asarray(temps[0])[:, None]], axis=-1)[None, ...]  # (1, N, 4), pair-i's own frame
    for i in range(n_steps):
        a_t = jnp.asarray(actions[i])[None, ...]  # (1, action_dims)
        delta_xyz_hat, delta_temp_hat = trainer.predict(x_t, a_t)
        x_next_xyz_local = x_t[0, :, :3] + delta_xyz_hat[0] / 100.0
        x_next_temp = x_t[0, :, 3] + delta_temp_hat[0, :, 0]

        x_next_xyz_world = untransform_points(x_next_xyz_local, jnp.asarray(quats[i]), jnp.asarray(positions[i]))
        pred_coords.append(np.asarray(x_next_xyz_world))
        pred_temps.append(np.asarray(x_next_temp))

        if i + 1 < n_steps:
            x_next_xyz_local_next = transform_points(x_next_xyz_world, jnp.asarray(quats[i + 1]), jnp.asarray(positions[i + 1]))
            x_t = jnp.concatenate([x_next_xyz_local_next, x_next_temp[:, None]], axis=-1)[None, ...]

    return np.stack(pred_coords), np.stack(pred_temps)


def _render_mesh_panel(pl, row, col, mesh_points, mesh_tetra, mesh_temps, bounds, title):
    cells = np.hstack([np.full((len(mesh_tetra), 1), 4, dtype=np.int64), mesh_tetra]).ravel()
    celltypes = np.full(len(mesh_tetra), pv.CellType.TETRA)
    grid = pv.UnstructuredGrid(cells, celltypes, mesh_points)
    grid["temperature_c"] = mesh_temps

    pl.subplot(row, col)
    pl.add_mesh(grid, scalars="temperature_c", cmap="inferno", clim=_CLIM, show_scalar_bar=False)
    pl.show_bounds(bounds=bounds, grid="back", location="outer", ticks="both", font_size=7)
    pl.add_axes(line_width=3)
    pl.view_isometric()
    pl.reset_camera(bounds=bounds)
    pl.camera_set = True
    pl.add_text(title, font_size=10)


def _render_point_panel(pl, row, col, points, temps, bounds, title):
    pl.subplot(row, col)
    cloud = pv.PolyData(points)
    cloud["temperature_c"] = temps
    pl.add_mesh(
        cloud, scalars="temperature_c", cmap="inferno", clim=_CLIM,
        style="points", render_points_as_spheres=True, point_size=5,
        show_scalar_bar=True, scalar_bar_args={"title": "T (C)", "fmt": "%.0f"},
    )
    pl.show_bounds(bounds=bounds, grid="back", location="outer", ticks="both", font_size=7)
    pl.add_axes(line_width=3)
    pl.view_isometric()
    pl.reset_camera(bounds=bounds)
    pl.camera_set = True
    pl.add_text(title, font_size=10)


def _render_frame(
    gt_mesh_points, gt_mesh_temps, gt_tetra, gt_sample_points, gt_sample_temps,
    pred_mesh_points, pred_mesh_temps, pred_tetra, pred_sample_points, pred_sample_temps,
    bounds, geom_errors, temp_errors, frame_idx, out_png: Path,
):
    pl = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(1000, 900))
    _render_mesh_panel(pl, 0, 0, gt_mesh_points, gt_tetra, gt_mesh_temps, bounds, f"ground truth -- real mesh (frame {frame_idx})")
    _render_point_panel(pl, 0, 1, gt_sample_points, gt_sample_temps, bounds, "ground truth -- sampled points (model input)")
    _render_mesh_panel(pl, 1, 0, pred_mesh_points, pred_tetra, pred_mesh_temps, bounds, "prediction -- LSQR-reconstructed mesh")
    _render_point_panel(pl, 1, 1, pred_sample_points, pred_sample_temps, bounds, "prediction -- sampled points (raw model output)")
    grid_png = out_png.with_suffix(".grid.png")
    pl.screenshot(str(grid_png))
    pl.close()

    n_steps = len(geom_errors)
    fig, ax_geom = plt.subplots(figsize=(4.5, 9))
    ax_geom.plot(range(n_steps), geom_errors, color="tab:blue", label="position error (mm)")
    ax_geom.set_xlabel("step")
    ax_geom.set_ylabel("mean position error (mm)", color="tab:blue")
    ax_geom.tick_params(axis="y", colors="tab:blue")
    ax_temp = ax_geom.twinx()
    ax_temp.plot(range(n_steps), temp_errors, color="tab:red", label="temperature error (C)")
    ax_temp.set_ylabel("mean |temperature error| (C)", color="tab:red")
    ax_temp.tick_params(axis="y", colors="tab:red")
    ax_geom.axvline(frame_idx, color="black", linestyle="--", alpha=0.7)
    ax_geom.set_title(f"rollout error -- step {frame_idx}/{n_steps - 1}")
    fig.tight_layout()
    error_png = out_png.with_suffix(".error.png")
    fig.savefig(error_png, dpi=110)
    plt.close(fig)

    from PIL import Image

    grid_img = Image.open(grid_png)
    error_img = Image.open(error_png)
    error_img = error_img.resize((int(error_img.width * grid_img.height / error_img.height), grid_img.height))
    composite = Image.new("RGB", (grid_img.width + error_img.width, grid_img.height), "white")
    composite.paste(grid_img, (0, 0))
    composite.paste(error_img, (grid_img.width, 0))
    composite.save(out_png)
    grid_png.unlink()
    error_png.unlink()


def render_series_thermal_gif(
    config: dict, trainer: ForgeNetTrainer, data: dict, series_idx: int = 0,
    out_path: "str | Path | None" = None, fps: int = 6, lsqr_alpha: float = 0.05,
):
    start, length = _pick_series(data, series_idx)
    series_id = str(data["series_ids"][series_idx])
    print(f"series_idx={series_idx}: series_id={series_id}, {length} pairs")

    coords = np.concatenate([data["coords_t"][start:start + 1], data["coords_tp1"][start:start + length]], axis=0)
    temps = np.concatenate([data["temp_t"][start:start + 1], data["temp_tp1"][start:start + length]], axis=0)
    action_features = config["network"]["action_features"]
    actions = actions_from_feature_map(action_features, {k: data[k][start:start + length] for k in action_features})
    # `phi` dropped from `action_features` (see process_data_forge_common.py's
    # per-hit canonicalization) but still stored raw for this exact purpose --
    # pair i's own WORLD -> CANONICAL quaternion, one per hit in this series.
    quats = np.stack([_phi_to_canonical_quat(phi) for phi in data["phi"][start:start + length]])
    # Real per-pair WORLD -> CANONICAL translation (zero unless the dataset
    # was built with `center_hit_z=True`) -- NOT assumed zero here, unlike
    # an earlier version of this function.
    positions = data["positions"][start:start + length]

    pred_coords, pred_temps = _rollout_series(trainer, coords, temps, actions, quats, positions)

    # `coords`/`temps` above are each pair's OWN canonical frame (pair i's
    # `coords_t`/`coords_tp1` share ONE frame with each other, not with
    # pair i-1/i+1's) -- world-frame them the same way `_rollout_series`
    # world-frames its output, so `coords_world`/`pred_coords` (world frame)
    # are directly comparable and consistent with the real (world-frame)
    # ground-truth mesh used below.
    coords_world = [untransform_points_np(coords[0], quats[0], positions[0])]
    coords_world += [untransform_points_np(coords[i], quats[i - 1], positions[i - 1]) for i in range(1, len(coords))]
    coords_world = np.stack(coords_world)

    geom_errors = np.mean(np.linalg.norm(pred_coords - coords_world, axis=-1), axis=-1)  # (n_steps+1,), mm
    temp_errors = np.mean(np.abs(pred_temps - temps), axis=-1)  # (n_steps+1,), degrees C

    # REAL recorded mesh, straight from the DB, for the ground-truth "mesh view".
    gt_points_real, gt_temps_real, tetra = _load_real_series_meshes(config["databases"]["db_path"], series_id)
    n_frames = coords.shape[0]
    if len(gt_points_real) != n_frames:
        raise ValueError(
            f"DB series {series_id} has {len(gt_points_real)} recorded steps, "
            f"but the npz pairing implies {n_frames} -- can't align real-mesh frames 1:1"
        )

    # LSQR reconstruction of the predicted mesh, one solve per step, warm-started
    # from the previous step's vertex deltas/temperatures for stability+speed.
    base_points = gt_points_real[0]  # rest state -- same convention `tet_ids[0]`/`bary_coords[0]` were sampled against
    tet_ids0 = data["tri_ids"][start].astype(int)
    bary_coords0 = data["bary_coords"][start]
    pred_mesh_points = [gt_points_real[0]]
    pred_mesh_temps = [gt_temps_real[0]]
    v_deltas_prev = None
    v_temp_prev = None
    for i in range(1, n_frames):
        v_deltas_prev, deformed_points = invert_deltas_to_tet_mesh(
            base_points, tetra, pred_coords[i], tet_ids0, bary_coords0,
            alpha=lsqr_alpha, initial_guess=v_deltas_prev,
        )
        v_temp_prev = invert_scalar_to_tet_mesh(
            base_points, tetra, pred_temps[i], tet_ids0, bary_coords0,
            alpha=lsqr_alpha, initial_guess=v_temp_prev,
        )
        pred_mesh_points.append(deformed_points)
        pred_mesh_temps.append(v_temp_prev)
        print(f"  LSQR reconstruction {i}/{n_frames - 1} done")

    all_pts = np.concatenate([np.concatenate(gt_points_real, axis=0), np.concatenate(pred_mesh_points, axis=0)], axis=0)
    bounds = (
        float(all_pts[:, 0].min()), float(all_pts[:, 0].max()),
        float(all_pts[:, 1].min()), float(all_pts[:, 1].max()),
        float(all_pts[:, 2].min()), float(all_pts[:, 2].max()),
    )

    out_path = Path(out_path) if out_path else Path(config["run"]["run_folder"]) / f"thermal_rollout_series{series_idx}.gif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / f"_frames_series{series_idx}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    for i in range(n_frames):
        frame_png = tmp_dir / f"frame_{i:04d}.png"
        _render_frame(
            gt_points_real[i], gt_temps_real[i], tetra, coords_world[i], temps[i],
            pred_mesh_points[i], pred_mesh_temps[i], tetra, pred_coords[i], pred_temps[i],
            bounds, geom_errors, temp_errors, i, frame_png,
        )
        frame_paths.append(frame_png)
        print(f"  frame {i + 1}/{n_frames} rendered (pos_err={geom_errors[i]:.3f}mm, temp_err={temp_errors[i]:.1f}C)")

    duration = 1000 / fps if fps > 0 else 200
    with imageio.get_writer(str(out_path), mode="I", duration=duration) as writer:
        for fp in frame_paths:
            writer.append_data(imageio.v2.imread(fp))
    for fp in frame_paths:
        fp.unlink()
    tmp_dir.rmdir()

    print(f"Saved -> {out_path}")
    return out_path


def evaluate_thermal(config: dict, trainer: ForgeNetTrainer, data: dict):
    """`eval.py`'s `evaluate()`, adapted for the temperature head -- SAME
    plotting functions (`compare_scatters`/`compare_vector_fields`/etc.,
    unchanged imports from `forge_net.utils.plotting`), called TWICE per
    `eval_idxs` entry: once with the ORIGINAL geometric `loss_cont` (squared
    position error, identical definition to `evaluate()`), once with
    `loss_cont` = squared temperature error, so both error signals get the
    same "existing point cloud plots" treatment (`compare_scatters_w_loss_
    cont`/`visualize_vector_diff_w_loss_cont`) rather than writing new plot
    code."""
    states = data["coords_t"]
    states_tp1 = data["coords_tp1"]
    temp_t = data["temp_t"]
    temp_tp1 = data["temp_tp1"]
    action_features = config["network"]["action_features"]
    actions = actions_from_feature_map(action_features, data)

    output_folder = Path(config["run"]["run_folder"])
    eval_path = output_folder / "eval_thermal"
    eval_path.mkdir(exist_ok=True, parents=True)

    for idx in config["eval"]["eval_idxs"]:
        idx_path = eval_path / str(idx)
        idx_path.mkdir(exist_ok=True)

        x_t_jax = jnp.concatenate(
            [jnp.array(states[idx]), jnp.array(temp_t[idx])[:, None]], axis=-1,
        )[jnp.newaxis, ...]  # (1, N, 4)
        a_jax = jnp.array(actions[idx])[jnp.newaxis, ...]

        delta_hat_jax, delta_temp_hat_jax = trainer.predict(x_t_jax, a_jax)
        delta_hat_jax = delta_hat_jax / 100.0

        x_t_np = np.array(states[idx])
        x_tp1_np = np.array(states_tp1[idx])
        delta_hat_np = jax.device_get(delta_hat_jax).squeeze(0)
        delta_temp_hat_np = jax.device_get(delta_temp_hat_jax).squeeze(0).squeeze(-1)

        x_tp1_hat = x_t_np + delta_hat_np
        delta_gt_np = x_tp1_np - x_t_np
        temp_delta_gt_np = np.array(temp_tp1[idx]) - np.array(temp_t[idx])

        # --- geometric error (identical definition to eval.py's evaluate()) ---
        geom_loss_cont = np.sum((delta_hat_np - delta_gt_np) ** 2, axis=1).squeeze()
        # --- temperature error, SAME per-point-scalar shape, reused by the SAME plotting fns ---
        temp_loss_cont = (delta_temp_hat_np - temp_delta_gt_np) ** 2

        compare_scatters(
            pc1=x_tp1_np, pc2=x_tp1_hat,
            label_1="Ground Truth Mesh Tp1", label_2="Predicted Mesh Tp1", label_3="G.T. vs. Predicted",
            fig_path=idx_path / f"compare_scatters_{idx}.png",
        )
        compare_scatters_w_loss_cont(
            pc1=x_t_np, pc2=x_tp1_hat, loss_cont=geom_loss_cont,
            fig_path=idx_path / f"compare_scatter_geom_error_{idx}.png",
        )
        compare_scatters_w_loss_cont(
            pc1=x_t_np, pc2=x_tp1_hat, loss_cont=temp_loss_cont,
            fig_path=idx_path / f"compare_scatter_temp_error_{idx}.png",
        )
        visualize_point_diff(
            x_tp1_hat, x_tp1_np, point_size=5,
            label="Vector Field Error \n(Predicted Deltas minus G.T. Deltas)",
            fig_path=idx_path / f"point_diff_{idx}.png",
        )
        compare_vector_fields(
            x_t_np, x_tp1_np, x_tp1_hat, min_magnitude=8.0,
            fig_path=idx_path / f"compare_vector_fields_{idx}.png",
        )
        visualize_vector_diff_w_scale(
            x_t=x_t_np, x_tp1=x_tp1_np, x_hat=x_tp1_hat, scale_factor=3.0,
            fig_path=idx_path / f"vector_fields_scale_{idx}.png",
        )
        visualize_vector_diff_w_loss_cont(
            x_t=x_t_np, x_tp1=x_tp1_np, x_hat=x_tp1_hat, loss_cont=geom_loss_cont,
            fig_path=idx_path / f"vector_fields_geom_error_{idx}.png",
        )
        visualize_vector_diff_w_loss_cont(
            x_t=x_t_np, x_tp1=x_tp1_np, x_hat=x_tp1_hat, loss_cont=temp_loss_cont,
            scale_vectors=True, scale_factor=10.0,  # magnitude-proportional (NOT fixed-length), sized up for visibility -- see visualize_vector_diff_w_loss_cont's docstring for the scale_vectors fix
            fig_path=idx_path / f"vector_fields_temp_error_{idx}.png",
        )
        plot_spherical_heatmap(
            vectors=x_tp1_np - x_tp1_hat, fig_path=idx_path / f"spherical_heatmap_{idx}.png",
        )

        print(f"idx={idx}: saved figures -> {idx_path} "
              f"(mean geom_err={geom_loss_cont.mean():.4f}, mean temp_err={temp_loss_cont.mean():.4f})")


def evaluate_series_thermal(
    config: dict, trainer: ForgeNetTrainer, data: dict, num_series: int, min_series_length: int,
    max_cols: int = 7, n_step: int = 3, fig_path: "Path | None" = None,
):
    """`eval.py`'s `evaluate_series`, adapted for a `predict_temperature=True`
    model -- mirrors its exact per-step one-step/recursive statistics
    collection and `plot_eval_series` call (same aggregate-over-many-series
    line chart, `mode='dist'`), with two differences:

    1. 4-channel input (`x_t = [coords, temp]`) and tuple model output
       (`delta_xyz, delta_temp = trainer.predict(...)`), instead of
       `evaluate_series`'s bare `delta = trainer.predict(...) / 100.0`
       (confirmed to crash on this model, see module docstring).
    2. Recursive-frame re-basing reuses `forge_net.utils.math.
       transform_points`/`untransform_points` with the npz's `rotations`/
       `positions` fields -- EXACTLY `evaluate_series`'s own
       `canonical_frame=True` branch, not a new mechanism, since
       `process_data_forge_common.py`'s per-hit canonicalization populates
       those fields with the REAL world<->canonical quaternion now (see
       `_phi_to_canonical_quat`), not the legacy schema's identity
       placeholder `evaluate_series` was originally built to skip via its
       `canonical_frame=False` branch.

    Temperature tracking is ADDITIVE: every stat `evaluate_series` already
    collects (`one_step`/`rec_step` MSE, dist mean/std/95th-pct) is
    collected UNCHANGED for position, plus the same shape of stat for
    |temperature error| (degrees C, not a vector distance -- a single
    signed-then-abs scalar per point, not `norm(...)`). No chamfer/
    hausdorff/mesh-reconstruction tracking (unlike `evaluate_series`'s
    optional `add_chamfer`/`add_hausdorff`) -- LSQR mesh reconstruction
    per step per series, for `num_series` series, would be a real cost this
    function doesn't need to pay just to add a temperature line to the
    chart."""
    states = data["coords_t"]
    states_tp1 = data["coords_tp1"]
    temp_t_arr = data["temp_t"]
    temp_tp1_arr = data["temp_tp1"]
    rotations = data["rotations"]
    positions = data["positions"]
    action_features = config["network"]["action_features"]
    actions = actions_from_feature_map(action_features, {k: data[k] for k in action_features})
    series_lengths = data["series_lengths"]

    output_folder = Path(config["run"]["run_folder"])
    eval_path = output_folder / "eval_series"
    eval_path.mkdir(exist_ok=True, parents=True)

    eval_series_idxs = [idx for idx, sl in enumerate(series_lengths) if sl >= min_series_length][:num_series]

    stat_keys = [
        "gt_steps", "one_step_preds", "rec_step_preds",
        "one_step_dist_means", "one_step_dist_stds", "one_step_dist_95pct_means", "one_step_dist_95pct_stds",
        "one_step_mses", "rec_step_dist_means", "rec_step_dist_stds", "rec_step_dist_95pct_means",
        "rec_step_dist_95pct_stds", "rec_step_mses",
        "one_step_temp_dist_means", "rec_step_temp_dist_means",
        # PER-POINT (not just mean) |temp error| arrays, same indexing as
        # one_step_preds/rec_step_preds -- lets the scatter panels color
        # predicted points by their own temperature error instead of a
        # flat color (see plot_eval_series's plot_pc).
        "one_step_temp_errors", "rec_step_temp_errors",
        # PER-POINT ACTUAL temperature (degrees C, not error) -- same
        # indexing as gt_steps/one_step_preds/rec_step_preds, lets
        # plot_eval_series color every point-cloud panel (GT included) by
        # the real thermal field, matching eval_thermal.py's own pyvista
        # renders (`_CLIM = (20.0, 1000.0)`, inferno) instead of leaving GT
        # panels flat-colored and predictions colored by error only.
        "gt_temps", "one_step_pred_temps", "rec_step_pred_temps",
    ]
    all_stats_dict = {f"all_{k}": [] for k in stat_keys}

    for eval_series_idx in tqdm(eval_series_idxs, desc="Evaluating Series (thermal)"):
        series_length = series_lengths[eval_series_idx]
        series_start_idx = sum(series_lengths[:eval_series_idx])
        series_end_idx = series_start_idx + series_length - 1 if series_length <= min_series_length else series_start_idx + min_series_length - 1

        series_stats_dict = {k: [] for k in stat_keys}

        # (1, N, 4) -- xyz + temp, matching the model's trained input shape.
        x_recursive_jax = jnp.concatenate(
            [jnp.array(states[series_start_idx]), jnp.array(temp_t_arr[series_start_idx])[:, None]], axis=-1,
        )[jnp.newaxis, ...]
        rec_temp = jnp.array(temp_t_arr[series_start_idx])  # (N,) -- carried forward separately, frame-invariant

        for idx in tqdm(range(series_start_idx, series_end_idx), leave=False):
            x_t_gt_np = states[idx]
            x_tp1_gt_np = states_tp1[idx]
            temp_t_gt_np = temp_t_arr[idx]
            temp_tp1_gt_np = temp_tp1_arr[idx]
            a_t_jax = jnp.array(actions[idx])[jnp.newaxis, ...]

            # One-step (always from real GT, both position and temp)
            x_t_gt_jax = jnp.concatenate(
                [jnp.array(x_t_gt_np), jnp.array(temp_t_gt_np)[:, None]], axis=-1,
            )[jnp.newaxis, ...]
            delta_xyz_one, delta_temp_one = trainer.predict(x_t_gt_jax, a_t_jax)
            x_tp1_hat_np = jax.device_get(jnp.array(x_t_gt_np) + delta_xyz_one[0] / 100.0)
            temp_tp1_hat_np = jax.device_get(jnp.array(temp_t_gt_np) + delta_temp_one[0, :, 0])

            # Recursive step (from the running state, carried forward)
            delta_xyz_rec, delta_temp_rec = trainer.predict(x_recursive_jax, a_t_jax)
            x_rec_xyz_local = x_recursive_jax[0, :, :3] + delta_xyz_rec[0] / 100.0
            rec_temp = rec_temp + delta_temp_rec[0, :, 0]
            x_rec_np = jax.device_get(x_rec_xyz_local)
            temp_rec_np = jax.device_get(rec_temp)

            # --- Position stats (identical shape/definition to evaluate_series) ---
            one_step_sq_diff = (x_tp1_hat_np - x_tp1_gt_np) ** 2
            one_step_dist_arr = np.linalg.norm(x_tp1_hat_np - x_tp1_gt_np, axis=-1)
            one_step_95pct = np.percentile(one_step_dist_arr, 95)
            one_step_95_arr = one_step_dist_arr[one_step_dist_arr >= one_step_95pct]

            series_stats_dict["one_step_mses"].append(np.mean(one_step_sq_diff))
            series_stats_dict["one_step_dist_means"].append(np.mean(one_step_dist_arr))
            series_stats_dict["one_step_dist_stds"].append(np.std(one_step_dist_arr))
            series_stats_dict["one_step_dist_95pct_means"].append(np.mean(one_step_95_arr))
            series_stats_dict["one_step_dist_95pct_stds"].append(np.std(one_step_95_arr))

            rec_step_sq_diff = (x_rec_np - x_tp1_gt_np) ** 2
            rec_step_dist_arr = np.linalg.norm(x_rec_np - x_tp1_gt_np, axis=-1)
            rec_step_95pct = np.percentile(rec_step_dist_arr, 95)
            rec_step_95_arr = rec_step_dist_arr[rec_step_dist_arr >= rec_step_95pct]

            series_stats_dict["rec_step_mses"].append(np.mean(rec_step_sq_diff))
            series_stats_dict["rec_step_dist_means"].append(np.mean(rec_step_dist_arr))
            series_stats_dict["rec_step_dist_stds"].append(np.std(rec_step_dist_arr))
            series_stats_dict["rec_step_dist_95pct_means"].append(np.mean(rec_step_95_arr))
            series_stats_dict["rec_step_dist_95pct_stds"].append(np.std(rec_step_95_arr))

            # --- Temperature stats (NEW -- absolute error, not a vector norm) ---
            one_step_temp_err_arr = np.abs(temp_tp1_hat_np - temp_tp1_gt_np)
            rec_step_temp_err_arr = np.abs(temp_rec_np - temp_tp1_gt_np)
            series_stats_dict["one_step_temp_dist_means"].append(np.mean(one_step_temp_err_arr))
            series_stats_dict["rec_step_temp_dist_means"].append(np.mean(rec_step_temp_err_arr))
            series_stats_dict["one_step_temp_errors"].append(one_step_temp_err_arr)
            series_stats_dict["rec_step_temp_errors"].append(rec_step_temp_err_arr)

            # Stats/dist arrays above are computed in pair idx's own LOCAL
            # (canonical) frame -- fine as-is, a shared rigid transform
            # cancels out of both a difference (translation) and a norm
            # (rotation), so world-framing wouldn't change them. But the
            # RAW arrays stored here feed plot_eval_series's scatter panels,
            # which overlay them against `gt_mesh_vertices_per_step`
            # (world-frame, straight from the DB) -- storing local-frame
            # points there was the bug: every step's scatter was rotated/
            # translated by that pair's own canonicalization relative to
            # the world-frame wireframe, so they never lined up. `states[idx]`/
            # `states_tp1[idx]` (and therefore `x_t_gt_np`/`x_tp1_hat_np`) are
            # BOTH in pair idx's own frame (`process_data_forge_common.py`
            # canonicalizes t and tp1 with the SAME quat/pos), and `x_rec_np`
            # is re-canonicalized into pair idx's frame at the end of the
            # previous iteration -- so `rotations[idx]`/`positions[idx]`
            # world-frames all three correctly.
            series_stats_dict["gt_steps"].append(untransform_points_np(x_t_gt_np, rotations[idx], positions[idx]))
            series_stats_dict["one_step_preds"].append(untransform_points_np(x_tp1_hat_np, rotations[idx], positions[idx]))
            series_stats_dict["rec_step_preds"].append(untransform_points_np(x_rec_np, rotations[idx], positions[idx]))
            series_stats_dict["gt_temps"].append(temp_t_gt_np)
            series_stats_dict["one_step_pred_temps"].append(temp_tp1_hat_np)
            series_stats_dict["rec_step_pred_temps"].append(temp_rec_np)

            # --- Recursive reference-frame update (position ONLY -- temp is
            # frame-invariant, `rec_temp` above already carries forward
            # unchanged) -- EXACTLY evaluate_series's canonical_frame=True
            # branch, reusing the real rotations/positions this dataset now
            # stores. ---
            if idx + 1 < series_end_idx:
                x_rec_world = untransform_points(x_rec_np, rotations[idx], positions[idx])
                x_rec_transformed = transform_points(x_rec_world, rotations[idx + 1], positions[idx + 1])
                x_recursive_jax = jnp.concatenate(
                    [x_rec_transformed, rec_temp[:, None]], axis=-1,
                )[jnp.newaxis, ...]

        for key in series_stats_dict:
            all_stats_dict[f"all_{key}"].append(series_stats_dict[key])

    # Real mesh for the LAST evaluated series (the one plot_eval_series'
    # scatter panels actually show, see its own docstring) -- straight from
    # the DB via the SAME helper render_series_thermal_gif already uses, so
    # the wireframe overlay reflects the actual recorded topology instead
    # of a convex-hull guess at it. `vertices_per_step[k]` aligns with
    # `all_gt_steps[-1][k]` -- both indexed by LOCAL pair position within
    # this series (gt_steps is appended in order starting fresh each
    # series' inner loop; _load_real_series_meshes returns rows in the
    # SAME `get_series_hits` step_number order).
    last_series_id = str(data["series_ids"][eval_series_idxs[-1]])
    gt_mesh_vertices_per_step, _, gt_mesh_tetra = _load_real_series_meshes(
        config["databases"]["db_path"], last_series_id,
    )

    fig_path = fig_path or (eval_path / "eval_series_thermal.png")
    plot_eval_series(
        all_stats_dict, mode="dist", max_cols=max_cols, n_step=n_step,
        fill_variation=True, add_temp=True,
        gt_mesh_vertices_per_step=gt_mesh_vertices_per_step, gt_mesh_tetra=gt_mesh_tetra,
        fig_path=fig_path,
    )

    print(f"\nEvaluation of {len(eval_series_idxs)} series (thermal) complete -> {fig_path}")
    return all_stats_dict


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--series-idx", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--evaluate", action="store_true", help="also run evaluate_thermal()'s per-idx scatter/vector-field plots")
    p.add_argument("--eval-series", action="store_true", help="also run evaluate_series_thermal()'s aggregate-over-many-series line chart (position + temp)")
    p.add_argument("--num-series", type=int, default=25, help="--eval-series only: how many series to aggregate over")
    p.add_argument("--min-series-length", type=int, default=15, help="--eval-series only: skip series shorter than this")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if not Path(config["run"]["run_folder"]).is_absolute():
        config["run"]["run_folder"] = str(get_project_root() / "runs" / config["run"]["run_name"])

    trainer, data = _load_trained_trainer(config)
    render_series_thermal_gif(config, trainer, data, series_idx=args.series_idx, out_path=args.out, fps=args.fps)

    if args.evaluate:
        evaluate_thermal(config, trainer, data)

    if args.eval_series:
        evaluate_series_thermal(config, trainer, data, num_series=args.num_series, min_series_length=args.min_series_length)


if __name__ == "__main__":
    main()
