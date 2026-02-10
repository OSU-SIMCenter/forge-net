import open3d as o3d
import numpy as np
from PIL import Image
import os
from forge_net.eval import evaluate_series

def poisson_recon_turntables(pc_data, output_path, n_frames=30):
    """
    Takes a numpy point cloud, reconstructs a surface, 
    and saves a turntable GIF.
    """
    # 1. Create PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_data)
    
    # 2. Estimate normals (required for Poisson reconstruction)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.65, max_nn=45))
    pcd.orient_normals_consistent_tangent_plane(10)

    # 3. Surface Reconstruction (Poisson)
    # depth=8 or 9 is usually a good balance of detail vs speed
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=15)
    
    # Cleanup: remove low-density vertices that create "bubbles" around the cloud
    # vertices_to_remove = densities < np.quantile(densities, 0.1)
    # mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7]) # A nice "Grey Clay" color

    # 4. Setup Off-screen Rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False) # Keep it hidden
    vis.add_geometry(mesh)
    
    # Center the camera on the mesh
    vis.get_view_control().set_zoom(0.8)
    
    images = []
    # 5. Rotate and capture frames
    for i in range(n_frames):
        # Rotate around the Y axis (up)
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0) # degrees
        vis.poll_events()
        vis.update_renderer()
        
        # Capture frame
        image = vis.capture_screen_float_buffer(do_render=True)
        images.append(Image.fromarray((np.asarray(image) * 255).astype(np.uint8)))

    vis.destroy_window()

    # 6. Save as GIF
    images[0].save(
        output_path, 
        save_all=True, 
        append_images=images[1:], 
        duration=100, 
        loop=0
    )
    print(f"Saved turntable to {output_path}")

if __name__ == "__main__":
    #Evaluate an existing trained model
    from model.trainer import Trainer
    from utils.utils import * 
    import yaml

    base_path = get_project_root()
    run_name = "mse_1024_unmasked_seeded"
    config_path = base_path / "runs" / run_name / "config_out.yml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    from main import make_dataloaders
    train_loader, test_loader = make_dataloaders(config)
    trainer = Trainer(config, train_loader, log_to_tb=False)
    
    recursive_preds = evaluate_series(config, trainer)
    output_folder = Path(config["run"]["run_folder"])
    eval_path = output_folder / "eval" / "surfaces"
    eval_path.mkdir(exist_ok=True)

    interval = 20
    for i, pc_np in enumerate(recursive_preds):
        if i % interval == 0:
            filename = f"surface_step_{i}.gif"
            poisson_recon_turntables(pc_np, eval_path / filename)