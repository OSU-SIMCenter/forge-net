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

Per-hit frame canonicalization (`_phi_to_canonical_quat`): every pair's
`coords_t`/`coords_tp1` are rotated about the axial axis by THAT pair's own
`phi` before sampling (via `transform_points_np`), so the strike always
lands at the same fixed reference direction (`phi=0`) in the frame the
network actually sees. This is why `"phi"` should typically be DROPPED from
a config's `action_features` once this is on -- it's now implicit in the
input/output geometry itself, not something the network needs told
directly. The exact transform used is stored into `rotations`/`positions`
(quaternion + zero translation, not a placeholder -- see that field's own
comment below), so a multi-hit consumer inverts it via `forge_net.utils.
math.untransform_points`/`transform_points`, the SAME mechanism `eval.py`'s
`evaluate_series` already uses for its own per-hit local frames -- not by
reconstructing a rotation from the raw `phi` value (also still stored, for
reference, but not the intended path).

Point sampling: three paths, dispatched per-pair off `mesh_snapshot_from_hit_row`'s
`cell_type` (see `forge_common.data_storage`'s schema docstring):
`forge_net.utils.math.tetrahedral_barycentric_sampling`/
`update_tetrahedral_barycentric_points` (volume-weighted, samples the
mesh's INTERIOR as well as its surface) when a hit's mesh has a volumetric
`tetra` connectivity (jax_forge/agility_forge_data always; cw_slab_model
when its adapter was built with `volumetric_mesh=True`, the
`forge_common.adapter_build.build_adapter` default used for all DB
recording); the surface-only `barycentric_sampling_np`/
`update_barycentric_points_np` for a `triangle`-only mesh; and, for
`cell_type='point_cloud'` sources with no connectivity at all (currently
just `genesis`'s raw MPM particles), `_sample_point_cloud` -- a direct
particle-INDEX subset, no barycentric interpolation, since MPM particles
ARE the physical material points already (no mesh to sample within). Point
correspondence across a (t, t+1) pair is entirely index-based in every
path (a sampled `(tet_id/tri_id, barycentric_coords)`, or for point_cloud a
raw particle index, at t is re-evaluated/re-indexed against t+1's OWN
array by that same index), so it's only valid when t and t+1 share
IDENTICAL connectivity (or, for point_cloud, the same particle count).

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

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

_FORGE_COMMON_ROOT = Path(__file__).resolve().parents[4] / "forge_common"
if str(_FORGE_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_FORGE_COMMON_ROOT))

from forge_common.data_storage import ForgeDBMS, mesh_snapshot_from_hit_row  # noqa: E402

from forge_net.utils.common import actions_from_feature_map  # noqa: E402
from forge_net.utils.math import (  # noqa: E402
    barycentric_sampling_np,
    tet_mask_from_window,
    tetrahedral_barycentric_sampling,
    transform_points_np,
    update_barycentric_points_np,
    update_tetrahedral_barycentric_points,
)

_ACTION_KEYS = ("rho", "phi", "z", "duration")
# `rotations`/`positions`: the REAL per-pair WORLD -> CANONICAL-frame
# transform (quaternion x,y,z,w + zero translation -- pure rotation, see
# `_phi_to_canonical_quat`), not a placeholder. Same field/convention
# `process_data.py`'s `process_series` already populates for its own
# `canonical_frame=True` path, and the SAME mechanism `eval.py`'s
# `evaluate_series` already uses (`transform_points`/`untransform_points`)
# to hop a recursive rollout's state from one pair's frame into the next
# pair's frame -- `eval_thermal.py`'s rollout reuses that exact pattern
# rather than a separate one built from raw `phi` values.
_ARRAY_KEYS = (
    "coords_t", "coords_tp1", "tri_ids", "bary_coords", "rotations", "positions", "cell_type",
    "temp_t", "temp_tp1",
) + _ACTION_KEYS
# `cell_type`: per-PAIR provenance ('tetra' | 'triangle' | 'point_cloud', from
# the pair's own mesh_snapshot, not a per-series constant -- kept alongside
# the array data specifically so a dataloader can key noise augmentation (or
# any other point_cloud-specific handling) off it without re-deriving it from
# raw coordinates. See dataloaders.py's `GetSingleStepDataLoaders`.
# `temp_t`/`temp_tp1`: per-vertex temperature (degrees C, `hits.
# input_temperature`) INTERPOLATED onto the SAME sampled points as
# `coords_t`/`coords_tp1`, via the SAME barycentric weights (see
# `tetrahedral_barycentric_sampling`'s/`update_tetrahedral_barycentric_
# points`'s existing `node_features` parameter -- built for exactly this,
# just unused until now). `cw_slab_model`/`cogging` data
# (`forge_common_thermal.db`) is currently the only source with real
# `input_temperature` -- pairs from any other source (or the triangle/
# point_cloud sampling paths, which don't thread `node_features` through
# yet) get an all-zero placeholder instead of `None`, so the output arrays
# stay a uniform shape regardless of provenance; a consumer that cares
# should filter on `source_sim`/`cell_type`, not assume every zero is real.


def _phi_to_canonical_quat(phi: float) -> np.ndarray:
    """The WORLD -> per-hit CANONICAL-frame rotation, as a quaternion (x,y,z,w
    -- scipy convention, matches `_IDENTITY_QUAT`/this module's `rotations`
    output field): rotates about the axial (X) axis so this hit's own `phi`
    always maps to the fixed reference direction `phi=0` (`forge_common.
    conventions`: `phi=0` -> +Y, positive `phi` rotates +Y toward +Z about
    +X). Every pair in a series gets rotated by ITS OWN hit's phi, so the
    network only ever needs to learn "what happens at phi=0" -- `phi` is
    dropped from `action_features` in the config once this is on (it's now
    implicit in the frame the network sees, not an explicit input).

    Sign: a point sitting at world-angle `phi` equals `R(phi) @ p_canonical`
    for the SAME point in canonical frame (`R(phi) @ (0,1,0) = (0, cos phi,
    sin phi)`, matching "positive phi rotates +Y toward +Z" directly), so
    recovering `p_canonical` from `p_world` is the INVERSE, `R(-phi)` --
    confirmed with a direct numeric check (a point built at `(x0, r*cos(phi),
    r*sin(phi))` must map back to `(x0, r, 0)` under this rotation), not just
    derived on paper. This SAME quaternion is stored into `out["rotations"]`
    below (with zero translation, `out["positions"]`) instead of the
    previous `_IDENTITY_QUAT`/`_ZERO_POSITION` placeholder -- so a consumer
    doing a multi-hit rollout can hop world -> this pair's canonical frame ->
    the NEXT pair's canonical frame via `forge_net.utils.math.
    transform_points`/`untransform_points`, exactly the pattern `eval.py`'s
    `evaluate_series` already uses for the legacy schema's per-hit local
    frames -- reuse that mechanism rather than inventing a parallel one (see
    `eval_thermal.py`'s `_rollout_series`)."""
    return Rotation.from_euler("x", -phi, degrees=False).as_quat()


def _mesh_to_pv_volume(mesh) -> "pv.UnstructuredGrid":
    """`MeshSnapshot` (`tetra` populated) -> a pure-TETRA `pv.UnstructuredGrid`,
    matching `forge_common.data_storage._tetra_boundary_faces`'s own
    construction (same cell-array layout the rest of forge_common already
    relies on)."""
    tetra = np.asarray(mesh.tetra, dtype=np.int64)
    cells = np.hstack([np.full((len(tetra), 1), 4, dtype=np.int64), tetra]).ravel()
    celltypes = np.full(len(tetra), pv.CellType.TETRA)
    return pv.UnstructuredGrid(cells, celltypes, np.asarray(mesh.vertices, dtype=np.float64))


def _boundary_sampling_weights(
    vertices: np.ndarray, tetra: np.ndarray, hit_z_mm: float,
    scale_mm: float = 10.0, boost: float = 4.0,
) -> np.ndarray:
    """Per-tet SOFT density multiplier for `tetrahedral_barycentric_sampling`'s
    `tet_weights` -- boosts sampling density near the press-contact region so
    the network sees more training signal (both input context and output
    supervision) exactly where deformation is sharpest, without excluding
    the rest of the mesh (see that function's `tet_weights` docstring).

    Cheap to define correctly here specifically BECAUSE frame canonicalization
    (`_phi_to_canonical_quat`) already ran on `vertices` before this is
    called: every pair's strike direction is rotated to the SAME fixed
    reference (`phi=0` -> +Y), so "near the press" is always "near the
    Y=0-ish plane at axial position `hit_z_mm`" in THIS mesh's frame --
    no per-pair angle bookkeeping needed. Distance is computed to the
    axial LINE `(x=hit_z_mm, z=0)` (dropping y), not a single point --
    the press contacts the workpiece across its full local radius at that
    axial band, not one fixed radius, so a "corridor" boost (Gaussian falloff
    in the x/z plane only) covers the actual contact surface rather than an
    arbitrary point on it.

    `scale_mm`: falloff width, same order of magnitude as a die's real
    axial footprint (`axial_halfwidth_mm=10.0` elsewhere in this project's
    thermal configs). `boost`: peak multiplier at the press location
    (weight -> `1+boost` there, decaying to `1` far away) -- moderate by
    design (not the "one region gets 100x the density" extreme a much
    smaller `scale_mm`/larger `boost` could produce), since the network
    still needs to see the WHOLE deforming shape, not just the contact
    edge."""
    centroids = vertices[tetra].mean(axis=1)  # (num_tets, 3)
    d2 = (centroids[:, 0] - hit_z_mm) ** 2 + centroids[:, 2] ** 2  # axial + off-strike-plane distance, y dropped
    return 1.0 + boost * np.exp(-d2 / (2.0 * scale_mm ** 2))


def _mesh_to_pv_surface(mesh) -> "pv.PolyData":
    """`MeshSnapshot` (surface-only) -> `pv.PolyData`."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_array = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), face_array)


def _sample_point_cloud(vertices: np.ndarray, total_points: int, seed=None) -> "tuple[np.ndarray, np.ndarray]":
    """Pick `total_points` particle INDICES from `vertices` (no
    connectivity/cells involved -- unlike the tetra/surface barycentric
    samplers, there's nothing to interpolate within, just a direct subset of
    real particles) -- without replacement if there are enough particles,
    with replacement otherwise (genesis's ~6.5k-particle clouds are well
    above any `total_points` used so far, so this is a rare-path safeguard,
    not the common case). Returns `(vertices[idx], idx)` -- the caller reuses
    `idx` to index the SAME particles at t+1, since MPM particle identity is
    stable across a hit (see caller)."""
    rng = np.random.default_rng(seed)
    n = len(vertices)
    idx = rng.choice(n, size=total_points, replace=n < total_points)
    return vertices[idx], idx


def _pair_seed(base_seed, series_id: str, step_number: int, per_pair_seed: bool = True) -> "int | None":
    """Deterministic PER-PAIR seed derived from the config's base seed +
    this series' id + this pair's step number -- NOT the same `base_seed`
    reused verbatim for every pair. `tetrahedral_barycentric_sampling`/
    `barycentric_sampling_np`/`_sample_point_cloud` all construct a FRESH
    `RandomState(seed)` per call (not an incrementally-advanced generator),
    so passing the literal same `base_seed` to every call (the previous
    behavior) reset the RNG to the identical starting draw sequence every
    single time -- the ONLY reason `coords_t` varies pair-to-pair was real
    geometry (mesh deformation) and, when `adaptive_sampling` is on, the
    hit-z-dependent tet weighting shifting which tets get sampled at all;
    the actual random DRAWS (barycentric position within whichever tet, the
    remainder tet assignment) were identical every call. Hashing in
    `series_id`/`step_number` gives every pair in the whole dataset its own
    independent sample instead. `coords_tp1` is UNCHANGED by this -- it
    still reuses `coords_t`'s own `cell_ids`/`bary_coords` via
    `update_tetrahedral_barycentric_points` (real point correspondence
    within a pair, not a second random draw), exactly as before.

    `per_pair_seed=False`: reverts to the OLD literal-reuse behavior
    (`base_seed` passed unchanged to every pair). Needed because independent
    per-pair sampling breaks point CORRESPONDENCE *across* a pair boundary --
    pair i's `coords_tp1` and pair i+1's `coords_t` represent the same real
    mesh state but are two separately-drawn point clouds, so a multi-hit
    rollout comparison (`eval_thermal.py`'s recursive tracking vs a fresh
    per-step ground-truth sample) sees an artificial jump at every pair
    boundary -- worse, and easy to misread as a real error, when
    `adaptive_sampling` is also on, since each pair's sample concentrates
    around ITS OWN (different) hit location. Keep this `False` ("fixed
    sampling") for practice/eval runs until that comparison is made
    correspondence-safe; flip back to `True` ("independent sampling", more
    diverse training signal) once verified. Both variants are worth keeping
    as separate `.npz` files rather than picking one -- see configs.

    `base_seed=None` (reproducibility opted out entirely) passes through as
    `None` regardless of `per_pair_seed` -- no derivation, matches the
    unseeded `RandomState()` path every sampler already falls back to."""
    if base_seed is None:
        return None
    if not per_pair_seed:
        return base_seed
    digest = hashlib.sha256(f"{series_id}:{step_number}".encode()).digest()
    offset = int.from_bytes(digest[:4], "big")
    return (base_seed + offset) % (2**32 - 1)


def process_series(
    series_id: str, rows: "list[dict]", total_points: int, seed=None,
    adaptive_sampling: bool = False, adaptive_sampling_scale_mm: float = 10.0, adaptive_sampling_boost: float = 4.0,
    window_sampling: bool = False, window_length_mm: float = 20.0, window_bc_length_mm: float = 10.0,
    center_hit_z: bool = False, per_pair_seed: bool = True,
) -> dict:
    """`rows`: this series' hit rows from `ForgeDBMS.get_series_hits`,
    already ordered by `step_number` (0 = initial pre-hit mesh, if
    present). Returns one dict of per-pair lists -- same field contract as
    `process_data.py`'s `process_series` (`coords_t`/`coords_tp1`/
    `tri_ids`/`bary_coords`) plus the new canonical action fields (`rho`/
    `phi`/`z`/`duration`, taken from the LATER hit in each pair -- that's
    the `Hit` that produced the `t -> t+1` transition) -- one entry per
    VALID consecutive `(t, t+1)` pair. See module docstring for exactly
    when a pair is skipped (missing mesh, or a topology change/remesh
    between the two rows).

    `adaptive_sampling`: when `True`, the TETRA sampling path (`cell_type
    == 'tetra'` -- triangle/point_cloud paths don't have per-tet volume
    density to weight, unaffected either way) boosts point density near
    the press-contact region (see `_boundary_sampling_weights`), rather
    than the previous uniform-by-volume-only density. `False` (default):
    unchanged, uniform-by-volume.

    `window_sampling`: HARD-restricts tetra sampling to tets within
    `window_length_mm` (+ `window_bc_length_mm` extra context) of this
    hit's own axial position (`tet_mask_from_window`, the tetra analog of
    `process_data.py`'s `triangle_mask_from_window`). NOT what makes `z`
    droppable -- kept as a separate, still-useful feature (e.g. local-patch
    training later), but confirmed the WRONG tool for frame-invariance: it
    EXCLUDES far-field material, which would sever the coupling between the
    localized PLASTIC deformation near the press and the surrounding
    (elastic/rigid-body-ish) far-field response the global-maxpool branch
    relies on seeing. Off by default; leave off unless a genuine local-patch
    use case comes up.

    `center_hit_z`: the ACTUAL mechanism for dropping `z` -- translates
    (not crops) every pair's `coords_t`/`coords_tp1` along the axial axis
    so THIS hit's own `z` always lands at the fixed reference `x=0`, exactly
    the same idea `_phi_to_canonical_quat` already applies to the angular
    DOF (a coordinate TRANSFORM, not a data restriction -- every point stays
    in the sample, just shifted). Combined with the existing rotation into
    one `transform_points_np` call (`canon_pos` below is no longer always
    zero)."""
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

        # Per-hit frame canonicalization: rotate BOTH sides of this pair
        # about the axial axis by THIS hit's own phi (row_tp1's -- "the Hit
        # that produced this transition", same hit whose rho/z/duration get
        # recorded below), so phi always maps to the same fixed reference
        # direction and can be dropped from the network's explicit inputs.
        # Rotation commutes with barycentric interpolation, so sampling the
        # ALREADY-rotated mesh below (rather than rotating the sampled
        # output points after the fact) is equivalent and simpler -- one
        # rotation of the full vertex array instead of one per sampled
        # point. Temperature is a per-vertex SCALAR, unaffected by rotating
        # positions -- `temp_t_full`/`temp_tp1_full` need no change. Reuses
        # `transform_points_np` (same rotation utility `eval.py`'s
        # `evaluate_series` already trusts) rather than hand-rolled matrix
        # math -- see `_phi_to_canonical_quat`.
        canon_quat = _phi_to_canonical_quat(row_tp1["phi"])
        # `center_hit_z`: translate along X (axial) so this hit's own z
        # lands at the fixed reference x=0 -- rotation is about X, so it
        # doesn't touch the X coordinate, meaning translation order (before
        # vs after the rotation applied below) doesn't matter here.
        canon_pos = np.array([-row_tp1["z"], 0.0, 0.0]) if center_hit_z else np.zeros(3)
        # `_boundary_sampling_weights`/`tet_mask_from_window` below operate
        # on `mesh_t.vertices` AFTER the canonicalizing transform (applied
        # a few lines down) -- when `center_hit_z` has already shifted the
        # hit's own z to x=0 in that transformed array, the window/weight
        # CENTER must be 0 too, not the raw world-frame `row_tp1["z"]`
        # (using the untranslated value here would be centering an
        # already-centered mesh on the wrong point).
        sampling_center_z = 0.0 if center_hit_z else row_tp1["z"]
        # This pair's own sampling seed -- per-pair-independent (default) or
        # the raw config `seed` reused verbatim across every pair
        # (`per_pair_seed=False`, "fixed sampling" -- see _pair_seed's
        # docstring for why each has its own use case right now).
        pair_seed = _pair_seed(seed, series_id, row_tp1["step_number"], per_pair_seed=per_pair_seed)
        mesh_t = dataclasses.replace(mesh_t, vertices=transform_points_np(mesh_t.vertices, canon_quat, canon_pos))
        mesh_tp1 = dataclasses.replace(mesh_tp1, vertices=transform_points_np(mesh_tp1.vertices, canon_quat, canon_pos))

        # Per-VERTEX temperature (degrees C), aligned index-for-index with
        # `mesh_t.vertices`/`mesh_tp1.vertices` -- `None` when this row has
        # no recorded temperature (e.g. a mechanical-only source), in which
        # case the sampling paths below fall back to an all-zero placeholder
        # (see `_ARRAY_KEYS`'s comment on `temp_t`/`temp_tp1`).
        temp_t_full = None if row_t["input_temperature"] is None else np.array(json.loads(row_t["input_temperature"]))
        temp_tp1_full = None if row_tp1["input_temperature"] is None else np.array(json.loads(row_tp1["input_temperature"]))

        if mesh_t.has_volume:
            cell_type_out = "tetra"
            grid_t = _mesh_to_pv_volume(mesh_t)
            grid_tp1 = _mesh_to_pv_volume(mesh_tp1)
            tet_mask = None
            if window_sampling:
                tet_mask = tet_mask_from_window(
                    mesh_t.vertices, mesh_t.tetra, center=sampling_center_z,
                    window_length=window_length_mm, bc_length=window_bc_length_mm,
                )
            tet_weights = None
            if adaptive_sampling:
                tet_weights = _boundary_sampling_weights(
                    mesh_t.vertices, mesh_t.tetra, hit_z_mm=sampling_center_z,
                    scale_mm=adaptive_sampling_scale_mm, boost=adaptive_sampling_boost,
                )
                if tet_mask is not None:
                    tet_weights = tet_weights[tet_mask]  # tetrahedral_barycentric_sampling indexes tet_weights against the ALREADY-masked tet list
            coords_t, cell_ids, bary_coords, temp_t = tetrahedral_barycentric_sampling(
                grid_t, total_points, seed=pair_seed, node_features=temp_t_full,
                tet_mask=tet_mask, tet_weights=tet_weights,
            )
            coords_tp1, temp_tp1 = update_tetrahedral_barycentric_points(
                grid_tp1, cell_ids, bary_coords, node_features=temp_tp1_full,
            )
        elif mesh_t.faces.size > 0:
            cell_type_out = "triangle"
            surf_t = _mesh_to_pv_surface(mesh_t)
            surf_tp1 = _mesh_to_pv_surface(mesh_tp1)
            coords_t, cell_ids, bary_coords = barycentric_sampling_np(surf_t, total_points, seed=pair_seed)
            coords_tp1 = update_barycentric_points_np(surf_tp1, cell_ids, bary_coords)
            temp_t = temp_tp1 = None  # surface-sampling path doesn't thread node_features through yet
        else:
            cell_type_out = "point_cloud"
            # point_cloud (e.g. genesis's raw MPM particles): no cell
            # connectivity to sample within, so no barycentric anything --
            # correspondence is direct particle-INDEX alignment instead (MPM
            # doesn't create/destroy/reorder particles mid-run, see
            # genesis_forge_adapter.py). Noise augmentation is applied later,
            # in the dataloader, not baked in here (same draw on both frames
            # of a pair so it perturbs the input without corrupting the
            # coords_tp1-coords_t delta label -- see dataloaders.py).
            if len(mesh_t.vertices) != len(mesh_tp1.vertices):
                print(f"  [{series_id}] skipping pair (step {row_t['step_number']}->{row_tp1['step_number']}): "
                      f"point_cloud particle count changed ({len(mesh_t.vertices)} -> {len(mesh_tp1.vertices)}) "
                      f"-- index correspondence assumption violated")
                continue
            coords_t, cell_ids = _sample_point_cloud(mesh_t.vertices, total_points, seed=pair_seed)
            coords_tp1 = mesh_tp1.vertices[cell_ids]
            bary_coords = np.zeros((total_points, 4), dtype=np.float64)  # not applicable, kept for shape consistency
            temp_t = None if temp_t_full is None else temp_t_full[cell_ids]
            temp_tp1 = None if temp_tp1_full is None else temp_tp1_full[cell_ids]

        if temp_t is None:
            temp_t = np.zeros(total_points, dtype=np.float64)
        if temp_tp1 is None:
            temp_tp1 = np.zeros(total_points, dtype=np.float64)

        out["coords_t"].append(coords_t)
        out["coords_tp1"].append(coords_tp1)
        out["tri_ids"].append(cell_ids)
        out["bary_coords"].append(bary_coords)
        out["cell_type"].append(cell_type_out)
        out["temp_t"].append(temp_t)
        out["temp_tp1"].append(temp_tp1)
        out["rotations"].append(tuple(canon_quat))
        out["positions"].append(tuple(canon_pos))
        out["rho"].append(row_tp1["rho"])
        out["phi"].append(row_tp1["phi"])
        out["z"].append(row_tp1["z"])
        out["duration"].append(row_tp1["duration"])

    return out


def _process_series_worker(args: tuple) -> "tuple[str, dict]":
    """`multiprocessing.Pool` target -- takes one `(series_id, rows,
    total_points, seed, adaptive_sampling, adaptive_sampling_scale_mm,
    adaptive_sampling_boost, window_sampling, window_length_mm,
    window_bc_length_mm, center_hit_z, per_pair_seed)` tuple (NOT separate
    args, so it's usable with `Pool.imap` like `process_data.py`'s own
    `process_series` worker) and returns `(series_id, result)`. Plain
    top-level function (not a closure/lambda) so it's picklable across the
    process boundary; `rows` is already a plain `list[dict]` (from
    `ForgeDBMS.get_series_hits`), not a live DB connection, so no DB access
    happens inside a worker."""
    (series_id, rows, total_points, seed, adaptive_sampling, adaptive_sampling_scale_mm, adaptive_sampling_boost,
     window_sampling, window_length_mm, window_bc_length_mm, center_hit_z, per_pair_seed) = args
    return series_id, process_series(
        series_id, rows, total_points, seed=seed,
        adaptive_sampling=adaptive_sampling,
        adaptive_sampling_scale_mm=adaptive_sampling_scale_mm,
        adaptive_sampling_boost=adaptive_sampling_boost,
        window_sampling=window_sampling,
        window_length_mm=window_length_mm,
        window_bc_length_mm=window_bc_length_mm,
        center_hit_z=center_hit_z,
        per_pair_seed=per_pair_seed,
    )


def extract_data(
    db_path: str,
    total_points: int,
    source_sims: "list[str] | None" = None,
    sequence_types: "list[str] | None" = None,
    seed=None,
    n_workers: "int | None" = None,
    adaptive_sampling: bool = False,
    adaptive_sampling_scale_mm: float = 10.0,
    adaptive_sampling_boost: float = 4.0,
    window_sampling: bool = False,
    window_length_mm: float = 20.0,
    window_bc_length_mm: float = 10.0,
    center_hit_z: bool = False,
    per_pair_seed: bool = True,
) -> dict:
    """Reads every matching series from `db_path` (all series if
    `source_sims`/`sequence_types` are `None`) and concatenates every
    series' valid pairs (see `process_series`) into flat arrays, mirroring
    `process_data.py`'s `n_extract_data` output contract INCLUDING its
    `multiprocessing.Pool` parallelization (`n_workers` here maps straight
    onto that) -- per-series sampling is independent and CPU-bound enough
    (tetrahedral barycentric sampling over a real mesh, per pair) that a
    few thousand series' worth is a real bottleneck serially. Every DB read
    (`get_all_series`/`get_series_hits`) still happens HERE, in the caller's
    process, before any worker spawns -- a live `ForgeDBMS`/`sqlite3`
    connection isn't picklable, so workers only ever see plain `list[dict]`
    row data, never the DB itself (mirrors `run_cogging_dataset_thermal.
    py`'s `_run_one_compute`/main-process-only-DB-access split).

    `n_workers=None` (default) or `<=1`: serial, in-process -- identical to
    this function's previous unconditional behavior, safe to call from
    ANYWHERE including a process that also needs the real GPU afterward
    (`train_forge_common.py`'s inline `make_dataset` call). `n_workers>1`:
    parallel via a `multiprocessing.Pool` in a `spawn` context -- ONLY safe
    to call from a process that doesn't need real GPU access itself or
    afterward, since `forge_net.utils.math` (imported by this module)
    imports `jax`, and `spawn` unconditionally re-imports the launching
    `__main__` module (and therefore this whole import chain, INCLUDING
    jax) in every worker before a `Pool(initializer=...)` callback ever
    gets to run -- there is no reliable way to force those workers onto
    CPU-only from inside this function. Callers that want parallel
    extraction must run it from a dedicated CPU-only entry point instead
    (see `build_dataset_forge_common.py`, which sets `CUDA_VISIBLE_DEVICES`
    before anything else and calls this with `n_workers>1`), then run
    `train_forge_common.py` normally -- `make_dataset` skips regeneration
    once the `.npz` already exists on disk, so this is a strict speedup,
    not a different code path."""
    db = ForgeDBMS(db_path, new=False)
    all_series = db.get_all_series()
    if source_sims is not None:
        all_series = [s for s in all_series if s["source_sim"] in source_sims]
    if sequence_types is not None:
        all_series = [s for s in all_series if s["sequence_type"] in sequence_types]

    series_rows = {}
    for series in all_series:
        rows = db.get_series_hits(series["id"])
        if len(rows) >= 2:
            series_rows[series["id"]] = rows
    db.close()

    output = {k: [] for k in _ARRAY_KEYS}
    output["series_lengths"] = []
    output["series_ids"] = []

    if n_workers is not None and n_workers > 1:
        import multiprocessing

        args_list = [
            (sid, rows, total_points, seed, adaptive_sampling, adaptive_sampling_scale_mm, adaptive_sampling_boost,
             window_sampling, window_length_mm, window_bc_length_mm, center_hit_z, per_pair_seed)
            for sid, rows in series_rows.items()
        ]
        print(f"Processing {len(args_list)} series using {n_workers} workers...")
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            results = pool.imap_unordered(_process_series_worker, args_list)
            for n_done, (series_id, result) in enumerate(results, 1):
                if n_done % 100 == 0 or n_done == len(args_list):
                    print(f"  ({n_done}/{len(args_list)} series processed)")
                n_pairs = len(result["coords_t"])
                if n_pairs == 0:
                    continue
                for key in _ARRAY_KEYS:
                    output[key].extend(result[key])
                output["series_lengths"].append(n_pairs)
                output["series_ids"].append(series_id)
    else:
        for series_id, rows in series_rows.items():
            print(f"Processing series {series_id} ({len(rows)} rows)...")
            result = process_series(
                series_id, rows, total_points, seed=seed,
                adaptive_sampling=adaptive_sampling,
                adaptive_sampling_scale_mm=adaptive_sampling_scale_mm,
                adaptive_sampling_boost=adaptive_sampling_boost,
                window_sampling=window_sampling,
                window_length_mm=window_length_mm,
                window_bc_length_mm=window_bc_length_mm,
                center_hit_z=center_hit_z,
                per_pair_seed=per_pair_seed,
            )
            n_pairs = len(result["coords_t"])
            if n_pairs == 0:
                continue
            for key in _ARRAY_KEYS:
                output[key].extend(result[key])
            output["series_lengths"].append(n_pairs)
            output["series_ids"].append(series_id)

    for key in ("coords_t", "coords_tp1", "tri_ids", "bary_coords", "rotations", "positions", "temp_t", "temp_tp1"):
        output[key] = np.array(output[key])
    for key in _ACTION_KEYS:
        output[key] = np.array(output[key], dtype=np.float64)
    # plain fixed-width unicode array (NOT dtype=object like series_ids below) --
    # a handful of short fixed strings, no need for allow_pickle=True on load
    output["cell_type"] = np.array(output["cell_type"])
    output["series_lengths"] = np.array(output["series_lengths"])
    output["series_ids"] = np.array(output["series_ids"], dtype=object)

    return output


def make_dataset(config: dict, n_workers: "int | None" = None):
    """Processes `forge_common.db` into a numpy `.npz` compatible with
    forge-net's pytorch dataloaders -- forge_common-schema counterpart to
    `process_data.py`'s `make_dataset`. `config["databases"]`:
    `{"db_path": ..., "source_sims": [...] | None, "sequence_types": [...] | None}`
    (a single forge_common db with a `source_sim` column to filter on,
    unlike `process_data.py`'s "concatenate exactly 2 separate db files"
    convention -- there's nothing to concatenate here, everything already
    lives in one db).

    `n_workers`: see `extract_data`'s docstring -- leave `None` (serial)
    when called from a process that also needs real GPU access (e.g.
    inline from `train_forge_common.py`); only pass `>1` from a dedicated
    CPU-only driver (`build_dataset_forge_common.py`).

    `config["datasets"]["adaptive_sampling"]` (bool, default `False`) +
    `"adaptive_sampling_scale_mm"`/`"adaptive_sampling_boost"` (see
    `_boundary_sampling_weights`) -- boundary-biased tetra sampling density,
    off by default so existing configs/datasets are unaffected.

    `config["datasets"]["window_sampling"]` (bool, default `False`) +
    `"window_length_mm"`/`"window_bc_length_mm"` (see `process_series`'s
    docstring / `tet_mask_from_window`) -- hard axial window, EXCLUDES
    far-field material. NOT what drops `z` from `action_features` (see
    `process_series`'s docstring for why -- severs the plastic/far-field
    coupling); kept as a separate, off-by-default feature.

    `config["datasets"]["center_hit_z"]` (bool, default `False`) -- the
    ACTUAL `z`-dropping mechanism: translates (not crops) every pair so
    this hit's own axial position lands at `x=0`, the same idea `phi`
    canonicalization already applies to the angular DOF.

    `config["datasets"]["per_pair_seed"]` (bool, default `True`) -- `False`
    ("fixed sampling") reuses the same literal `seed` for every pair
    instead of independently seeding each one (see `_pair_seed`'s
    docstring) -- needed for a correspondence-safe multi-hit rollout
    comparison in `eval_thermal.py` until that comparison is made
    correspondence-safe some other way."""
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
        n_workers=n_workers,
        adaptive_sampling=ds_cfg.get("adaptive_sampling", False),
        adaptive_sampling_scale_mm=ds_cfg.get("adaptive_sampling_scale_mm", 10.0),
        adaptive_sampling_boost=ds_cfg.get("adaptive_sampling_boost", 4.0),
        window_sampling=ds_cfg.get("window_sampling", False),
        window_length_mm=ds_cfg.get("window_length_mm", 20.0),
        window_bc_length_mm=ds_cfg.get("window_bc_length_mm", 10.0),
        center_hit_z=ds_cfg.get("center_hit_z", False),
        per_pair_seed=ds_cfg.get("per_pair_seed", True),
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
    agnostic, just a tensor-shape/backend adapter).

    `config["datasets"]["point_cloud_noise_std"]` (mm, default 0.0 -- off):
    forwarded to `GetSingleStepDataLoaders`/`SingleStepMeshTransitionDataset`
    together with `data["cell_type"]`, so `cell_type='point_cloud'` pairs
    (currently just `genesis`) get the same-noise-both-frames jitter
    per-epoch in the dataloader -- see that class's docstring for why it has
    to be the SAME draw on both frames of a pair.

    Returns `(train_loader, test_loader, loss_norm_stats)` -- the 3rd
    element is new: `{"pos_delta_std": float, "temp_delta_std": float}`,
    the std of the (already `delta_scalar`-scaled) position delta and the
    (unscaled) temperature delta across the WHOLE dataset, computed once
    here from the actual data rather than guessed -- `ForgeNetTrainer`'s
    `predict_temperature` loss combines the two loss terms normalized by
    these (see `trainer.py`'s `_get_loss_fn`), since raw position-delta and
    temperature-delta MSEs live on very different natural scales (sub-mm
    vs. tens-to-hundreds of degrees C) and would otherwise have one term
    silently dominate the combined loss."""
    from forge_net.data.dataloaders import GetSingleStepDataLoaders
    from forge_net.data.process_data import jax_dataloader_wrapper

    data_path = config["datasets"]["data_out"]
    data = np.load(data_path)
    c_t = data["coords_t"]
    c_tp1 = data["coords_tp1"]
    temp_t = data["temp_t"]
    temp_tp1 = data["temp_tp1"]

    action_features = config["network"]["action_features"]
    actions = actions_from_feature_map(action_features, data)

    delta_scalar = config["network"].get("delta_scalar", 100.0)
    train_loader_pt, test_loader_pt = GetSingleStepDataLoaders(
        coords_t=c_t, coords_tp1=c_tp1, actions=actions,
        batch_size=config["network"]["batch_size"],
        cell_type=data["cell_type"],
        point_cloud_noise_std=config["datasets"].get("point_cloud_noise_std", 0.0),
        temp_t=temp_t, temp_tp1=temp_tp1, delta_scalar=delta_scalar,
    )
    train_loader = lambda: jax_dataloader_wrapper(train_loader_pt)  # noqa: E731
    test_loader = lambda: jax_dataloader_wrapper(test_loader_pt)  # noqa: E731

    loss_norm_stats = {
        "pos_delta_std": float(np.std(delta_scalar * (c_tp1 - c_t))),
        "temp_delta_std": float(np.std(temp_tp1 - temp_t)),
    }
    return train_loader, test_loader, loss_norm_stats


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
