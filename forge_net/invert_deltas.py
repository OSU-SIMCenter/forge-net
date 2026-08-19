import os
import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr
from PIL import Image
from pathlib import Path
# from forge_net.eval import evaluate_series

def get_graph_laplacian(mesh):
    num_vertices = mesh.n_points
    faces = mesh.regular_faces

    v1, v2, v3 = faces[:, 0], faces[:, 1], faces[:, 2]
    rows = np.concatenate([v1, v2, v2, v3, v3, v1])
    cols = np.concatenate([v2, v1, v3, v2, v1, v3])

    data = np.ones_like(rows, dtype=float)
    W = coo_matrix((data, (rows, cols)), shape=(num_vertices, num_vertices)).tocsr()
    W.data[:] = 1.0

    degree = np.array(W.sum(axis=1)).flatten()
    D = coo_matrix((degree, (np.arange(num_vertices), np.arange(num_vertices))),
                   shape=(num_vertices, num_vertices)).tocsr()
    return D - W


def get_tet_graph_laplacian(num_vertices, tetra):
    """`get_graph_laplacian`'s tet-mesh analog -- vertex adjacency from all 6
    edges of every tetrahedron (`regular_faces`-based triangle adjacency has
    no tet equivalent), otherwise identical construction (unweighted
    graph Laplacian D - W)."""
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    rows, cols = [], []
    for i, j in pairs:
        rows.append(tetra[:, i]); cols.append(tetra[:, j])
        rows.append(tetra[:, j]); cols.append(tetra[:, i])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    data = np.ones_like(rows, dtype=float)
    W = coo_matrix((data, (rows, cols)), shape=(num_vertices, num_vertices)).tocsr()
    W.data[:] = 1.0

    degree = np.array(W.sum(axis=1)).flatten()
    D = coo_matrix((degree, (np.arange(num_vertices), np.arange(num_vertices))),
                   shape=(num_vertices, num_vertices)).tocsr()
    return D - W


def _build_tet_sample_matrix(num_vertices, tetra, tet_ids, bary_coords):
    """Shared by `invert_deltas_to_tet_mesh`/`invert_scalar_to_tet_mesh` --
    the sparse (num_samples, num_vertices) barycentric-weight matrix `A`
    such that `A @ vertex_field ~= sampled_field` for ANY per-vertex field
    (positions, temperature, ...), plus the Jacobi column-scaling used to
    balance the LSQR solve (see `invert_deltas_to_mesh`'s steps 3-4, same
    math, just factored out so the tet position/scalar variants don't
    duplicate it). Returns `(A_scaled, M_inv, relevant_indices)`."""
    num_samples = tet_ids.shape[0]
    relevant_indices = tetra[tet_ids]  # (num_samples, 4)

    rows = np.repeat(np.arange(num_samples), 4)
    cols = relevant_indices.flatten()
    weights = bary_coords.flatten()
    A = coo_matrix((weights, (rows, cols)), shape=(num_samples, num_vertices)).tocsr()

    col_norms = np.sqrt(np.array(A.power(2).sum(axis=0))).flatten()
    col_norms[col_norms == 0] = 1.0
    M_inv = coo_matrix((1.0 / col_norms, (np.arange(num_vertices), np.arange(num_vertices))),
                       shape=(num_vertices, num_vertices))
    A_scaled = A @ M_inv
    return A_scaled, M_inv, col_norms, relevant_indices


def invert_deltas_to_tet_mesh(base_points, tetra, predicted_pc, tet_ids, bary_coords,
                               alpha=0.05, initial_guess=None):
    """`invert_deltas_to_mesh`'s tet-mesh analog -- that function hard-codes
    a TRIANGLE surface mesh (3 barycentric weights per sample, `base_mesh.
    faces`); `cw_slab_model`'s (and any other volumetric source's) sampled
    points are barycentric combinations of a TET's 4 vertices instead (see
    `process_data_forge_common.py`'s `bary_coords` shape comment -- K=4 for
    `cell_type='tetra'`), so reusing it directly on tet data would silently
    mis-reshape `bary_coords` (`(-1, 3)`) and mis-index `tetra[tri_ids]`
    against a 3-column-only formula. Same LSQR-per-axis approach otherwise
    (Jacobi column scaling, optional graph-Laplacian regularization, damp
    =0.1, `initial_guess` warm start) -- see that function's inline
    comments for the step-by-step rationale, identical here.

    Args:
        base_points: (V, 3) REST-pose vertex positions this reconstruction
            solves DELTAS relative to -- pass the mesh's OWN true rest
            state (e.g. a series' step-0 real mesh from the DB), not an
            arbitrary intermediate frame, so `tet_ids`/`bary_coords` (which
            must be sampled relative to THIS SAME rest configuration) stay
            geometrically consistent hit-to-hit across a whole rollout.
        tetra: (K, 4) int, this mesh's fixed tet connectivity.
        predicted_pc: (num_samples, 3) the model's predicted sample
            positions to fit vertex deltas to.
        tet_ids, bary_coords: (num_samples,), (num_samples, 4) -- SAME
            sampling used to produce `predicted_pc`'s training pair (from
            `tetrahedral_barycentric_sampling`/the saved npz `tri_ids`/
            `bary_coords`), so `A @ base_points + A @ v_deltas` reproduces
            `predicted_pc` in the least-squares sense.

    Returns `(v_deltas, deformed_points)` -- `(V, 3)` reconstructed vertex
    DELTAS and `base_points + v_deltas`.
    """
    predicted_pc = np.asarray(predicted_pc).reshape(-1, 3)
    tet_ids = np.asarray(tet_ids).astype(int).flatten()
    bary_coords = np.asarray(bary_coords).reshape(-1, 4)
    base_points = np.asarray(base_points)
    num_vertices = base_points.shape[0]

    A_scaled, M_inv, col_norms, relevant_indices = _build_tet_sample_matrix(
        num_vertices, tetra, tet_ids, bary_coords
    )

    v0 = base_points[relevant_indices[:, 0]]
    v1 = base_points[relevant_indices[:, 1]]
    v2 = base_points[relevant_indices[:, 2]]
    v3 = base_points[relevant_indices[:, 3]]
    p_rest = (bary_coords[:, 0:1] * v0 + bary_coords[:, 1:2] * v1
              + bary_coords[:, 2:3] * v2 + bary_coords[:, 3:4] * v3)
    point_deltas = predicted_pc - p_rest

    if alpha > 0:
        L = get_tet_graph_laplacian(num_vertices, tetra)
        L_scaled = L @ M_inv
        A_final = vstack([A_scaled, np.sqrt(alpha) * L_scaled])
        rhs_final = np.vstack([point_deltas, np.zeros((num_vertices, 3))])
    else:
        A_final = A_scaled
        rhs_final = point_deltas

    x0_scaled = initial_guess * col_norms[:, np.newaxis] if initial_guess is not None else None

    v_deltas_scaled = np.zeros((num_vertices, 3))
    for d in range(3):
        x0_d = x0_scaled[:, d] if x0_scaled is not None else None
        res = lsqr(A_final, rhs_final[:, d], damp=0.1, x0=x0_d)
        v_deltas_scaled[:, d] = res[0]

    v_deltas = v_deltas_scaled / col_norms[:, np.newaxis]
    return v_deltas, base_points + v_deltas


def invert_scalar_to_tet_mesh(base_points, tetra, predicted_scalar, tet_ids, bary_coords,
                               alpha=0.05, initial_guess=None):
    """`invert_deltas_to_tet_mesh`'s scalar-field analog (e.g. per-vertex
    temperature, degrees C) -- SAME sparse system/LSQR approach, but no
    "rest pose" to subtract (a scalar field has no rest/deformed
    distinction the way position does), so this solves DIRECTLY for
    absolute per-vertex values: `A @ v_scalar ~= predicted_scalar`, one
    LSQR solve instead of 3.

    Returns `(V,)` reconstructed per-vertex scalar values.
    """
    predicted_scalar = np.asarray(predicted_scalar).reshape(-1)
    tet_ids = np.asarray(tet_ids).astype(int).flatten()
    bary_coords = np.asarray(bary_coords).reshape(-1, 4)
    num_vertices = base_points.shape[0] if base_points is not None else int(tetra.max()) + 1

    A_scaled, M_inv, col_norms, _ = _build_tet_sample_matrix(num_vertices, tetra, tet_ids, bary_coords)

    if alpha > 0:
        L = get_tet_graph_laplacian(num_vertices, tetra)
        L_scaled = L @ M_inv
        A_final = vstack([A_scaled, np.sqrt(alpha) * L_scaled])
        rhs_final = np.concatenate([predicted_scalar, np.zeros(num_vertices)])
    else:
        A_final = A_scaled
        rhs_final = predicted_scalar

    x0_scaled = initial_guess * col_norms if initial_guess is not None else None
    res = lsqr(A_final, rhs_final, damp=0.1, x0=x0_scaled)
    v_scalar_scaled = res[0]
    return v_scalar_scaled / col_norms

def invert_deltas_to_mesh(base_mesh, predicted_pc, tri_ids, bary_coords, 
                          alpha=0.05, initial_guess=None):
    
    # 1. Shape Standardization
    predicted_pc = np.asarray(predicted_pc).reshape(-1, 3)
    tri_ids = np.asarray(tri_ids).astype(int).flatten()
    bary_coords = np.asarray(bary_coords).reshape(-1, 3)
    
    num_samples = predicted_pc.shape[0]
    num_vertices = base_mesh.n_points

    if base_mesh.faces.size > 0:
        triangles = base_mesh.faces.reshape(-1, 4)[:, 1:]
    else:
        raise ValueError("Base mesh has no faces.")

    # 2. Calculate Target Point Deltas (RHS)
    v_rest = base_mesh.points
    relevant_indices = triangles[tri_ids]
    
    v0 = v_rest[relevant_indices[:, 0]]
    v1 = v_rest[relevant_indices[:, 1]]
    v2 = v_rest[relevant_indices[:, 2]]
    
    p_rest = (bary_coords[:, 0:1] * v0 + 
              bary_coords[:, 1:2] * v1 + 
              bary_coords[:, 2:3] * v2)
    
    point_deltas = predicted_pc - p_rest

    # 3. Build Sparse System Matrix A
    rows = np.repeat(np.arange(num_samples), 3)
    cols = relevant_indices.flatten()
    weights = bary_coords.flatten()
    
    A = coo_matrix((weights, (rows, cols)), shape=(num_samples, num_vertices)).tocsr()

    # 4. Column Scaling (Jacobi Preconditioning)
    # Scale columns to have unit norm to balance solver gradients
    col_norms = np.sqrt(np.array(A.power(2).sum(axis=0))).flatten()
    col_norms[col_norms == 0] = 1.0 
    
    M_inv = coo_matrix((1.0 / col_norms, (np.arange(num_vertices), np.arange(num_vertices))),
                       shape=(num_vertices, num_vertices))
    
    A_scaled = A @ M_inv

    # 5. Regularization & Stacking
    if alpha > 0:
        L = get_graph_laplacian(base_mesh)
        L_scaled = L @ M_inv # Scale Laplacian to match A
        
        A_final = vstack([A_scaled, np.sqrt(alpha) * L_scaled])
        rhs_padding = np.zeros((num_vertices, 3))
        rhs_final = np.vstack([point_deltas, rhs_padding])
    else:
        A_final = A_scaled
        rhs_final = point_deltas

    # 6. Warm Start Scaling
    x0_scaled = None
    if initial_guess is not None:
        # Transform previous deltas into the scaled coordinate system
        x0_scaled = initial_guess * col_norms[:, np.newaxis]

    # 7. Solve
    v_deltas_scaled = np.zeros((num_vertices, 3))
    for d in range(3):
        x0_d = x0_scaled[:, d] if x0_scaled is not None else None
        res = lsqr(A_final, rhs_final[:, d], damp=0.1, x0=x0_d)
        v_deltas_scaled[:, d] = res[0]
        
    # 8. Unscale and Apply
    v_deltas = v_deltas_scaled / col_norms[:, np.newaxis]
    
    deformed_mesh = base_mesh.copy()
    deformed_mesh.points += v_deltas
    
    return deformed_mesh, v_deltas

def save_comparison_turntable(pred_mesh, gt_mesh, output_path, n_frames=150, fps=12):
    """
    Creates a side-by-side GIF comparing predicted vs ground truth mesh.
    """
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[1920, 1080])
    
    # Subplot 0: Prediction
    plotter.subplot(0, 0)
    plotter.add_text("Network Prediction", font_size=12)
    plotter.add_mesh(pred_mesh, color="lightblue", show_edges=True)
    
    # Subplot 1: Ground Truth
    plotter.subplot(0, 1)
    plotter.add_text("Ground Truth", font_size=12)
    plotter.add_mesh(gt_mesh, color="gray", show_edges=True)
    
    plotter.link_views() # Synchronize camera movements
    plotter.camera_position = 'iso'
    plotter.reset_camera()
    center = pred_mesh.center

    path = plotter.generate_orbital_path(n_points=n_frames, shift=pred_mesh.length, viewup=[0, 0, 1])
    
    plotter.open_gif(str(output_path), fps=fps)
    for pos in path.points:
        plotter.camera_position = [pos, center, [0, 0, 1]]
        
        plotter.render()
        plotter.write_frame()
    
    plotter.close()
    print(f"Comparison saved to {output_path}")

if __name__ == "__main__":

    from forge_net.model.trainer import ForgeNetTrainer
    from forge_net.eval import evaluate_series
    from forge_net.utils.common import * 
    import yaml

    base_path = get_project_root()
    run_name = "chamfer_1024_unmasked_seeded_tri_ids"
    config_path = base_path / "runs" / run_name / "config_out.yml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    from main import make_dataloaders
    train_loader, test_loader = make_dataloaders(config)
    trainer = ForgeNetTrainer(config, train_loader, log_to_tb=False)
    
    data_path = config["datasets"]["data_out"]
    assert os.path.exists(data_path), "Dataset found"
    data  = np.load(data_path, allow_pickle=True)
    
    recursive_preds, sidx, eidx = evaluate_series(config, trainer)
    output_folder = Path(config["run"]["run_folder"])
    eval_path = output_folder / "eval" / "surfaces"
    eval_path.mkdir(exist_ok=True)

    mesh_data = data['meshes'][0]
    base_mesh_pv = pv.PolyData(mesh_data['points'], mesh_data['faces'])
    tri_ids = data['tri_ids']
    bary_coords = data['bary_coords']
    gt_meshes = data['meshes_tp1']
    print(len(gt_meshes[0]['faces']))
    interval = 20
    for i, pc_np in enumerate(recursive_preds):
        if i % interval == 0:
            # 1. Recover the mesh vertices from point cloud predictions
            recovered_mesh = invert_deltas_to_mesh(
                base_mesh_pv, pc_np, tri_ids[i], bary_coords[i], alpha=0
            )
            
            # 2. Get the corresponding Ground Truth mesh
            gt_mesh_pv = pv.PolyData(gt_meshes[i])
            
            # 3. Generate side-by-side comparison
            filename = f"comparison_step_{i}.gif"
            save_comparison_turntable(recovered_mesh, gt_mesh_pv, eval_path / filename)