"""Process `forge_common.db`'s unified `series`/`hits` schema into
forgenet's single-step training `.npz` format -- the forge_common-schema
counterpart to `process_data.py` (reads the older `strike` table, a
different schema entirely) and `process_data_fanglei.py` (targets a
similar-but-not-identical schema and is currently broken -- undefined
`s_tp1`, a stray `raise()` in its own `make_dataloaders`).

Reads via `forge_common.data_storage.ForgeDBMS`/`get_series_hits`/
`mesh_snapshot_from_hit_row` -- the clean Python API forge_common already
built for this, instead of hand-rolling SQL/JSON parsing like the other two
scripts. That also means every mesh reconstruction (surface-vs-volume,
tetra-vs-triangle, mesh_topologies dedup) is handled identically to every
other forge_common consumer (`forge_common.viz`, `render_db_replay_grid.py`,
etc.) -- no second implementation to drift out of sync.

Action space: `forge_common.hit.Hit(rho, phi, z, duration)` -- a single
canonical, physically-meaningful action shared across ALL THREE simulators
(jax_forge/agility_forge_data/cw_slab_model), unlike this repo's existing
action features (`utils/common.py`'s `actions_from_feature_map`: raw
solver-step count, or Agility-only X-position/roll-degrees). `rho`/`z` are
mm, `phi` is RADIANS (raw, not sin/cos -- the press is double-sided, so a
strike at `phi` is physically identical to one at `phi+pi`; every phi this
data generator produces stays well within a single ~pi-wide span with no
real wraparound to worry about), `duration` is seconds. See
`forge_net.utils.common.actions_from_feature_map`'s new `"rho"`/`"phi"`/
`"z"`/`"duration"` entries.

Point sampling: `forge_net.utils.math.tetrahedral_barycentric_sampling`/
`update_tetrahedral_barycentric_points` (volume-weighted, samples the
mesh's INTERIOR as well as its surface) when a hit's mesh has a volumetric
`tetra` connectivity (jax_forge/agility_forge_data always; cw_slab_model
when its adapter was built with `volumetric_mesh=True`, the
`forge_common.adapter_build.build_adapter` default used for all DB
recording) -- falls back to the surface-only `barycentric_sampling_np`/
`update_barycentric_points_np` for a `triangle`-only mesh. Point
correspondence across a (t, t+1) pair is entirely index-based (a sampled
`(tet_id/tri_id, barycentric_coords)` at t is re-evaluated against t+1's
OWN cell array by that same index), so it's only valid when t and t+1
share IDENTICAL connectivity.

Topology-change ("remesh") handling: `jax_forge`/`agility_forge_data` can
re-tetrahedralize at the end of any `apply_hit` call if mesh quality
degrades (`remesh_min_quality`) -- never mid-hit, but that means hit t+1's
OWN stored connectivity can differ from hit t's even though hit t+1's
solve itself ran entirely on hit t's topology. `forge_common`'s
`topology_hash` (content-hash-deduplicated per row) makes this directly
checkable with no extra bookkeeping: any consecutive pair whose
`topology_hash` differs is skipped (logged, not silently sampled into
garbage correspondences) -- the row AFTER the remesh remains a perfectly
valid anchor for the NEXT pair, only the single pair spanning the change is
lost. `cw_slab_model` never remeshes (fixed lattice for its whole
lifetime), so this never triggers for it.

`step_number=0` (a series' initial, pre-hit mesh -- see
`forge_common.data_storage.ForgeDBMS.record_initial_mesh`) is just another
row here: the pairing loop naturally includes `(step 0, step 1)` as this
series' first training pair when present, recovering the first hit's own
transition (older series recorded before `record_initial_mesh` existed
simply start their first pair at `(step 1, step 2)` instead -- no crash,
just one fewer pair for that series).

Usage (from `models/forge-net`):
    python -m forge_net.data.process_data_forge_common  # see __main__ below for a config example
"""

import sys
from pathlib import Path

import numpy as np
import pyvista as pv

_FORGE_COMMON_ROOT = Path(__file__).resolve().parents[4] / "forge_common"
if str(_FORGE_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_FORGE_COMMON_ROOT))

from forge_common.data_storage import ForgeDBMS, mesh_snapshot_from_hit_row  # noqa: E402

from forge_net.utils.common import actions_from_feature_map  # noqa: E402
from forge_net.utils.math import (  # noqa: E402
    barycentric_sampling_np,
    tetrahedral_barycentric_sampling,
    update_barycentric_points_np,
    update_tetrahedral_barycentric_points,
)

_ACTION_KEYS = ("rho", "phi", "z", "duration")
# `rotations`/`positions`: forge-net's `eval.py` (`evaluate_series`) uses
# these as a per-hit LOCAL reference-frame quaternion+translation to
# re-align a recursive rollout's prediction into the next hit's own frame
# (`untransform_points`/`transform_points`) -- a real need for the legacy
# schema, whose stored points are transformed into PRESS-LOCAL coordinates
# per hit (see `process_data.py:58-59`). forge_common's stored vertices are
# already in ONE canonical global frame for a whole series (the entire
# point of forge_common's unified action space), so there is no per-hit
# transform to record -- identity rotation `(0,0,0,1)` (scipy xyzw) and
# zero translation are the LITERAL truth here, not a placeholder, and let
# `evaluate_series` run against this data completely unmodified.
_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
_ZERO_POSITION = (0.0, 0.0, 0.0)
_ARRAY_KEYS = ("coords_t", "coords_tp1", "tri_ids", "bary_coords", "rotations", "positions") + _ACTION_KEYS


def _mesh_to_pv_volume(mesh) -> "pv.UnstructuredGrid":
    """`MeshSnapshot` (`tetra` populated) -> a pure-TETRA `pv.UnstructuredGrid`,
    matching `forge_common.data_storage._tetra_boundary_faces`'s own
    construction (same cell-array layout the rest of forge_common already
    relies on)."""
    tetra = np.asarray(mesh.tetra, dtype=np.int64)
    cells = np.hstack([np.full((len(tetra), 1), 4, dtype=np.int64), tetra]).ravel()
    celltypes = np.full(len(tetra), pv.CellType.TETRA)
    return pv.UnstructuredGrid(cells, celltypes, np.asarray(mesh.vertices, dtype=np.float64))


def _mesh_to_pv_surface(mesh) -> "pv.PolyData":
    """`MeshSnapshot` (surface-only) -> `pv.PolyData`."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_array = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), face_array)


def process_series(series_id: str, rows: "list[dict]", total_points: int, seed=None) -> dict:
    """`rows`: this series' hit rows from `ForgeDBMS.get_series_hits`,
    already ordered by `step_number` (0 = initial pre-hit mesh, if
    present). Returns one dict of per-pair lists -- same field contract as
    `process_data.py`'s `process_series` (`coords_t`/`coords_tp1`/
    `tri_ids`/`bary_coords`) plus the new canonical action fields (`rho`/
    `phi`/`z`/`duration`, taken from the LATER hit in each pair -- that's
    the `Hit` that produced the `t -> t+1` transition) -- one entry per
    VALID consecutive `(t, t+1)` pair. See module docstring for exactly
    when a pair is skipped (missing mesh, or a topology change/remesh
    between the two rows)."""
    out = {k: [] for k in _ARRAY_KEYS}

    for i in range(len(rows) - 1):
        row_t, row_tp1 = rows[i], rows[i + 1]
        if row_t["vertices"] is None or row_tp1["vertices"] is None:
            print(f"  [{series_id}] skipping pair (step {row_t['step_number']}->{row_tp1['step_number']}): "
                  f"no mesh on one side")
            continue
        if row_t["topology_hash"] != row_tp1["topology_hash"]:
            print(f"  [{series_id}] skipping pair (step {row_t['step_number']}->{row_tp1['step_number']}): "
                  f"topology changed (remesh) -- no valid point correspondence")
            continue

        mesh_t = mesh_snapshot_from_hit_row(row_t)
        mesh_tp1 = mesh_snapshot_from_hit_row(row_tp1)

        if mesh_t.has_volume:
            grid_t = _mesh_to_pv_volume(mesh_t)
            grid_tp1 = _mesh_to_pv_volume(mesh_tp1)
            coords_t, cell_ids, bary_coords, _ = tetrahedral_barycentric_sampling(
                grid_t, total_points, seed=seed
            )
            coords_tp1, _ = update_tetrahedral_barycentric_points(grid_tp1, cell_ids, bary_coords)
        else:
            surf_t = _mesh_to_pv_surface(mesh_t)
            surf_tp1 = _mesh_to_pv_surface(mesh_tp1)
            coords_t, cell_ids, bary_coords = barycentric_sampling_np(surf_t, total_points, seed=seed)
            coords_tp1 = update_barycentric_points_np(surf_tp1, cell_ids, bary_coords)

        out["coords_t"].append(coords_t)
        out["coords_tp1"].append(coords_tp1)
        out["tri_ids"].append(cell_ids)
        out["bary_coords"].append(bary_coords)
        out["rotations"].append(_IDENTITY_QUAT)
        out["positions"].append(_ZERO_POSITION)
        out["rho"].append(row_tp1["rho"])
        out["phi"].append(row_tp1["phi"])
        out["z"].append(row_tp1["z"])
        out["duration"].append(row_tp1["duration"])

    return out


def extract_data(
    db_path: str,
    total_points: int,
    source_sims: "list[str] | None" = None,
    sequence_types: "list[str] | None" = None,
    seed=None,
) -> dict:
    """Reads every matching series from `db_path` (all series if
    `source_sims`/`sequence_types` are `None`) and concatenates every
    series' valid pairs (see `process_series`) into flat arrays, mirroring
    `process_data.py`'s `n_extract_data` output contract (minus the
    multiprocessing -- forge_common series are typically few enough, and
    `cw_slab_model` cheap enough, that serial processing is fine; revisit
    if `jax_forge`/`agility_forge_data` volumes grow large)."""
    db = ForgeDBMS(db_path, new=False)
    all_series = db.get_all_series()
    if source_sims is not None:
        all_series = [s for s in all_series if s["source_sim"] in source_sims]
    if sequence_types is not None:
        all_series = [s for s in all_series if s["sequence_type"] in sequence_types]

    output = {k: [] for k in _ARRAY_KEYS}
    output["series_lengths"] = []
    output["series_ids"] = []

    for series in all_series:
        rows = db.get_series_hits(series["id"])
        if len(rows) < 2:
            continue
        print(f"Processing series {series['id']} ({series['source_sim']}, {series['sequence_type']}, "
              f"{series['status']}, {len(rows)} rows)...")
        result = process_series(series["id"], rows, total_points, seed=seed)
        n_pairs = len(result["coords_t"])
        if n_pairs == 0:
            continue
        for key in _ARRAY_KEYS:
            output[key].extend(result[key])
        output["series_lengths"].append(n_pairs)
        output["series_ids"].append(series["id"])

    db.close()

    for key in ("coords_t", "coords_tp1", "tri_ids", "bary_coords", "rotations", "positions"):
        output[key] = np.array(output[key])
    for key in _ACTION_KEYS:
        output[key] = np.array(output[key], dtype=np.float64)
    output["series_lengths"] = np.array(output["series_lengths"])
    output["series_ids"] = np.array(output["series_ids"], dtype=object)

    return output


def make_dataset(config: dict):
    """Processes `forge_common.db` into a numpy `.npz` compatible with
    forge-net's pytorch dataloaders -- forge_common-schema counterpart to
    `process_data.py`'s `make_dataset`. `config["databases"]`:
    `{"db_path": ..., "source_sims": [...] | None, "sequence_types": [...] | None}`
    (a single forge_common db with a `source_sim` column to filter on,
    unlike `process_data.py`'s "concatenate exactly 2 separate db files"
    convention -- there's nothing to concatenate here, everything already
    lives in one db)."""
    ds_cfg = config["datasets"]
    total_points = ds_cfg["total_points"]
    seed = ds_cfg.get("seed")
    data_out = ds_cfg["data_out"]

    if Path(data_out).exists():
        print(f"{data_out} already exists, skipping creation")
        return

    db_cfg = config["databases"]
    db_path = db_cfg["db_path"]
    assert Path(db_path).exists(), f"forge_common db not found at {db_path}"

    data = extract_data(
        db_path, total_points,
        source_sims=db_cfg.get("source_sims"),
        sequence_types=db_cfg.get("sequence_types"),
        seed=seed,
    )
    Path(data_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(data_out, **data)
    print(f"Saved dataset -> {data_out} ({len(data['coords_t'])} pairs from {len(data['series_ids'])} series)")


def make_dataloaders(config: dict):
    """Same contract as `process_data.py`'s `make_dataloaders` -- loads the
    `.npz` `make_dataset` wrote, builds the action tensor via
    `actions_from_feature_map(config["network"]["action_features"], data)`
    (now able to use `"rho"`/`"phi"`/`"z"`/`"duration"`, see module
    docstring), and wraps `GetSingleStepDataLoaders` in the existing JAX
    wrapper (reused as-is from `process_data.py` -- it's dataset-schema
    agnostic, just a tensor-shape/backend adapter)."""
    from forge_net.data.dataloaders import GetSingleStepDataLoaders
    from forge_net.data.process_data import jax_dataloader_wrapper

    data_path = config["datasets"]["data_out"]
    data = np.load(data_path)
    c_t = data["coords_t"]
    c_tp1 = data["coords_tp1"]

    action_features = config["network"]["action_features"]
    actions = actions_from_feature_map(action_features, data)

    train_loader_pt, test_loader_pt = GetSingleStepDataLoaders(
        coords_t=c_t, coords_tp1=c_tp1, actions=actions,
        batch_size=config["network"]["batch_size"],
    )
    train_loader = lambda: jax_dataloader_wrapper(train_loader_pt)  # noqa: E731
    test_loader = lambda: jax_dataloader_wrapper(test_loader_pt)  # noqa: E731
    return train_loader, test_loader


if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parents[4]
    example_config = {
        "databases": {
            "db_path": str(_ROOT / "forge_common" / "data" / "forge_common.db"),
            "source_sims": ["cw_slab_model"],
            "sequence_types": ["cogging"],
        },
        "datasets": {
            "total_points": 2048,
            "seed": 0,
            "data_out": str(Path(__file__).parent / "datasets" / "cw_slab_model_cogging.npz"),
        },
        "network": {
            "action_features": ["rho", "phi", "z"],
            "batch_size": 8,
        },
    }
    make_dataset(example_config)
