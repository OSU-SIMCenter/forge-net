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
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[1024, 512])
    
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

    from model.trainer import Trainer
    from utils.utils import * 
    import yaml

    base_path = get_project_root()
    run_name = "mse_1024_unmasked_seeded_tri_ids"
    config_path = base_path / "runs" / run_name / "config_out.yml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    from main import make_dataloaders
    train_loader, test_loader = make_dataloaders(config)
    trainer = Trainer(config, train_loader, log_to_tb=False)
    
    data_path = config["datasets"]["data_out"]
    assert os.path.exists(data_path), "Dataset found"
    data  = np.load(data_path, allow_pickle=True)
    
    # recursive_preds, sidx, eidx = evaluate_series(config, trainer)s
    output_folder = Path(config["run"]["run_folder"])
    eval_path = output_folder / "eval" / "surfaces"
    eval_path.mkdir(exist_ok=True)

    mesh_data = data['meshes'][0]
    base_mesh_pv = pv.PolyData(mesh_data['points'], mesh_data['faces'])
    tri_ids = data['tri_ids']
    bary_coords = data['bary_coords']
    gt_meshes = data['meshes_tp1']
    print(len(gt_meshes[0]['faces']))
    # interval = 20
    # for i, pc_np in enumerate(recursive_preds):
    #     if i % interval == 0:
    #         # 1. Recover the mesh vertices from point cloud predictions
    #         recovered_mesh = invert_deltas_to_mesh(
    #             base_mesh_pv, pc_np, tri_ids[i], bary_coords[i], alpha=0
    #         )
            
    #         # 2. Get the corresponding Ground Truth mesh
    #         gt_mesh_pv = pv.PolyData(gt_meshes[i])
            
    #         # 3. Generate side-by-side comparison
    #         filename = f"comparison_step_{i}.gif"
    #         save_comparison_turntable(recovered_mesh, gt_mesh_pv, eval_path / filename)