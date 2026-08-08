from pathlib import Path
import torch
from forge_net.utils.common import *
from forge_net.utils.math import *
from forge_net.utils.plotting import * 
from forge_net.invert_deltas import invert_deltas_to_mesh, save_comparison_turntable
from forge_net.loss.chamfer_jax import chamfer_distance_jax
# from pytorch3d.loss import chamfer_distance #original implementation uses a chamfer distance
from tqdm import tqdm 
import jax
import jax.numpy as jnp
import flax.linen as nn

def evaluate(config, trainer):
    '''
    Evaluate a trained ForgeNet network by creating some simple scatter and vector plots
    '''
    data_path = config["datasets"]["data_out"]
    assert os.path.exists(data_path), "Dataset found"
    data = np.load(data_path)
    
    # JAX native shape is (N, 3), no need to transpose yet
    states = data['coords_t']
    states_tp1 = data['coords_tp1']
    action_features = config["network"]["action_features"]
    
    actions = actions_from_feature_map(action_features, data)
    
    output_folder = Path(config["run"]["run_folder"])
    eval_path = output_folder / "eval"
    eval_path.mkdir(exist_ok=True)
        
    for idx in config["eval"]["eval_idxs"]:
        idx_path = eval_path / str(idx)
        idx_path.mkdir(exist_ok=True)

        # Prepare JAX inputs: (1, N, 3) and (1, Dims)
        x_t_jax = jnp.array(states[idx])[jnp.newaxis, ...]
        a_jax = jnp.array(actions[idx])[jnp.newaxis, ...]
        
        # 1. Inference using our new JAX trainer
        # This returns a JAX Array on the GPU/TPU
        delta_hat_jax = trainer.predict(x_t_jax, a_jax) / 100.0
        
        # 2. Convert to NumPy immediately for plotting compatibility
        # jax.device_get is the safe way to pull data to CPU NumPy
        x_t_np = np.array(states[idx])
        x_tp1_np = np.array(states_tp1[idx])
        delta_hat_np = jax.device_get(delta_hat_jax).squeeze(0)
        
        # Calculate prediction in coordinate space
        x_tp1_hat = x_t_np + delta_hat_np
        
        # Ground truth delta
        delta_gt_np = x_tp1_np - x_t_np

        # 3. Handle Loss calculations for the plots
        # We temporarily convert to Torch only for the loss function if needed
        if config["network"]["loss"] == "mse":
            loss_cont = np.sum((delta_hat_np - delta_gt_np)**2, axis=1).squeeze()
        
        elif config["network"]["loss"] in ["chamfer", "wsd"]:
            d_gt_jax = jnp.array(delta_gt_np)[jnp.newaxis, ...]
            d_hat_jax = jnp.array(delta_hat_np)[jnp.newaxis, ...]

            loss_cont = chamfer_distance_jax(d_gt_jax, d_hat_jax,
                                              point_reduction=None,
                                              batch_reduction=None).loss[0]
            loss_cont = np.array(loss_cont).squeeze()
        
        else:
            raise ValueError("Loss function not supported for evaluation")

        # --- Your existing plotting functions work as-is now ---
        compare_scatters(pc1=x_tp1_np, pc2=x_tp1_hat, 
                    label_1="Ground Truth Mesh Tp1", 
                    label_2="Predicted Mesh Tp1", 
                    label_3='G.T. vs. Predicted',
                    fig_path= idx_path / f"compare_scatters_{idx}.png")
        
        compare_scatters_w_loss_cont(pc1=x_t_np, pc2=x_tp1_hat, loss_cont=loss_cont, 
                                     fig_path=idx_path / f"compare_scatter_loss_{idx}.png")

        visualize_point_diff(x_tp1_hat, x_tp1_np, point_size=5, 
                             label="Vector Field Error \n(Predicted Deltas minus G.T. Deltas)",
                             fig_path = idx_path / f"point_diff_{idx}.png")
        
        compare_vector_fields(x_t_np, x_tp1_np, x_tp1_hat, min_magnitude=8.0, 
                              fig_path=idx_path / f"compare_vector_fields_{idx}.png")
        
        visualize_vector_diff_w_scale(x_t=x_t_np, x_tp1=x_tp1_np, x_hat=x_tp1_hat,
                                        scale_factor=3.0,
                                        fig_path = idx_path / f"vector_fields_scale_{idx}.png")
        
        visualize_vector_diff_w_loss_cont(x_t=x_t_np, x_tp1=x_tp1_np, x_hat=x_tp1_hat, 
                                            loss_cont=loss_cont,
                                            fig_path = idx_path / f"vector_fields_loss_{idx}.png")
        
        plot_spherical_heatmap(vectors=x_tp1_np - x_tp1_hat, 
                                            fig_path = idx_path / f"spherical_heatmap_{idx}.png" )

        print(f"Saved figures in {idx_path}")

def evaluate_series(config, trainer, num_series, min_series_length,
                    max_cols, plot_mode, n_step, add_mse, add_chamfer,
                    add_hausdorff, plot_heatmaps, save_meshes):
    # `process_series` (see data/process_data.py) canonicalizes BOTH coords_t
    # AND coords_tp1 for row i via `transform_points(points, rotations[i],
    # positions[i])` -- but ONLY when `canonical_frame=True`. When
    # `canonical_frame=False` (our world-frame slab datasets), `coords_t`/
    # `coords_tp1` are stored in raw world frame directly and
    # `positions`/`rotations` are just action-feature VALUES, not a
    # canonicalizing pose -- so the untransform/transform frame-hop below
    # (needed to move the recursive rollout state from hit i's local frame
    # into hit i+1's local frame) must be skipped entirely, or it corrupts
    # the rollout with a spurious rigid-body shift every single hit.
    canonical_frame = config["datasets"].get("canonical_frame", True)

    data_path = config["datasets"]["data_out"]
    assert os.path.exists(data_path), "Dataset found"
    data = np.load(data_path, allow_pickle=True)
    
    # Raw data is usually (N, 3)
    states = data['coords_t']
    states_tp1 = data['coords_tp1']
    rotations = data['rotations']
    positions = data['positions']
    action_features = config["network"]["action_features"]
    actions = actions_from_feature_map(action_features, data)
    series_lengths = data['series_lengths']

    output_folder = Path(config["run"]["run_folder"])
    eval_path = output_folder / "eval_series"
    eval_path.mkdir(exist_ok=True, parents=True)

    eval_series_idxs = [idx for idx, sl in enumerate(series_lengths) if sl >= min_series_length][:num_series]
    
    # Initialize all_stats_dict with exactly the same keys as the original
    all_stats_dict = {f'all_{k}': [] for k in [
        'gt_steps', 'one_step_preds', 'rec_step_preds',
        'one_step_dist_means', 'one_step_dist_stds', 'one_step_dist_95pct_means', 'one_step_dist_95pct_stds',
        'one_step_mses', 'rec_step_dist_means', 'rec_step_dist_stds', 'rec_step_dist_95pct_means',
        'rec_step_dist_95pct_stds', 'rec_step_mses', 'rec_step_chamfers', 'rec_step_hausdorffs',
        'rec_step_chamfer_to_last', 'rec_step_mse_to_last'
    ]}

    for eval_series_idx in tqdm(eval_series_idxs, desc="Evaluating Series"):
        series_length = series_lengths[eval_series_idx]
        series_start_idx = sum(series_lengths[:eval_series_idx])
        # Match the original truncation logic
        series_end_idx = series_start_idx + series_length - 1 if series_length <= min_series_length else series_start_idx + min_series_length - 1
        last_frame_idx = series_end_idx - 1

        series_stats_dict = {k: [] for k in [
            'gt_steps', 'one_step_preds', 'rec_step_preds',
            'one_step_dist_means', 'one_step_dist_stds', 'one_step_dist_95pct_means', 'one_step_dist_95pct_stds',
            'one_step_mses', 'rec_step_dist_means', 'rec_step_dist_stds', 'rec_step_dist_95pct_means',
            'rec_step_dist_95pct_stds', 'rec_step_mses', 'rec_step_chamfers', 'rec_step_hausdorffs',
            'rec_step_chamfer_to_last', 'rec_step_mse_to_last'
        ]}

        # Initialize recursive state: Shape (1, N, 3) for JAX
        x_recursive_jax = jnp.array(states[series_start_idx])[jnp.newaxis, ...]
        previous_deltas = None
        counter = 0

        for idx in tqdm(range(series_start_idx, series_end_idx), leave=False):
            # 1. Inputs
            x_t_gt_np = states[idx]
            x_tp1_gt_np = states_tp1[idx]
            a_t_jax = jnp.array(actions[idx])[jnp.newaxis, ...]
            
            # 2. JAX Inference
            # One-step (always from GT)
            x_t_gt_jax = jnp.array(x_t_gt_np)[jnp.newaxis, ...]
            delta_one_jax = trainer.predict(x_t_gt_jax, a_t_jax) / 100.0
            x_tp1_hat_np = jax.device_get(x_t_gt_jax + delta_one_jax).squeeze(0)
            
            # Recursive step
            delta_rec_jax = trainer.predict(x_recursive_jax, a_t_jax) / 100.0
            x_recursive_jax = x_recursive_jax + delta_rec_jax
            x_rec_np = jax.device_get(x_recursive_jax).squeeze(0)

            # 3. Stats Calculation (matching your original NumPy logic exactly)
            # One-Step stats
            one_step_sq_diff = (x_tp1_hat_np - x_tp1_gt_np)**2
            one_step_dist_arr = np.linalg.norm(x_tp1_hat_np - x_tp1_gt_np, axis=-1)
            one_step_95pct = np.percentile(one_step_dist_arr, 95)
            one_step_95_arr = one_step_dist_arr[one_step_dist_arr >= one_step_95pct]

            series_stats_dict['one_step_mses'].append(np.mean(one_step_sq_diff))
            series_stats_dict['one_step_dist_means'].append(np.mean(one_step_dist_arr))
            series_stats_dict['one_step_dist_stds'].append(np.std(one_step_dist_arr))
            series_stats_dict['one_step_dist_95pct_means'].append(np.mean(one_step_95_arr))
            series_stats_dict['one_step_dist_95pct_stds'].append(np.std(one_step_95_arr))

            # Recursive stats
            rec_step_sq_diff = (x_rec_np - x_tp1_gt_np)**2
            rec_step_dist_arr = np.linalg.norm(x_rec_np - x_tp1_gt_np, axis=-1)
            rec_step_95pct = np.percentile(rec_step_dist_arr, 95)
            rec_step_95_arr = rec_step_dist_arr[rec_step_dist_arr >= rec_step_95pct]

            series_stats_dict['rec_step_mses'].append(np.mean(rec_step_sq_diff))
            series_stats_dict['rec_step_dist_means'].append(np.mean(rec_step_dist_arr))
            series_stats_dict['rec_step_dist_stds'].append(np.std(rec_step_dist_arr))
            series_stats_dict['rec_step_dist_95pct_means'].append(np.mean(rec_step_95_arr))
            series_stats_dict['rec_step_dist_95pct_stds'].append(np.std(rec_step_95_arr))

            # 4. Save Point Clouds (N, 3)
            series_stats_dict['gt_steps'].append(x_t_gt_np)
            series_stats_dict['one_step_preds'].append(x_tp1_hat_np)
            series_stats_dict['rec_step_preds'].append(x_rec_np)

            # 5. Complex Metrics (Chamfer/Hausdorff)
            if add_chamfer:
                rec_jax = jnp.array(x_rec_np)[jnp.newaxis, ...]
                gt_jax = jnp.array(x_tp1_gt_np)[jnp.newaxis, ...]
                series_stats_dict['rec_step_chamfers'].append(
                    float(chamfer_distance_jax(rec_jax, gt_jax).loss))

                # Chamfer to Last Frame Goal
                if canonical_frame:
                    x_rec_world = untransform_points(x_rec_np, rotations[idx], positions[idx])
                    x_rec_in_last = transform_points(x_rec_world, rotations[last_frame_idx], positions[last_frame_idx])
                else:
                    x_rec_in_last = x_rec_np  # already world frame -- no frame-hop needed

                rec_last_jax = jnp.array(x_rec_in_last)[jnp.newaxis, ...]
                last_gt_jax = jnp.array(states_tp1[last_frame_idx])[jnp.newaxis, ...]
                series_stats_dict['rec_step_chamfer_to_last'].append(
                    float(chamfer_distance_jax(rec_last_jax, last_gt_jax).loss))

            if add_hausdorff:
                # Mesh logic uses standard NumPy/PyVista workflow
                mesh_data = data['meshes'][0]
                base_mesh = pv.PolyData(mesh_data['points'], mesh_data['faces'])
                recovered_mesh, current_deltas = invert_deltas_to_mesh(
                    base_mesh, x_rec_np, data['tri_ids'][idx], data['bary_coords'][idx], 
                    alpha=0.05, initial_guess=previous_deltas
                )
                previous_deltas = current_deltas
                gt_mesh_pv = pv.PolyData(data['meshes_tp1'][idx]['points'], mesh_data['faces'])
                
                series_stats_dict['rec_step_hausdorffs'].append(compute_haussdorff_distance(gt_mesh_pv, recovered_mesh))
                
                if save_meshes and (counter + 1) % n_step == 0:
                    save_comparison_turntable(recovered_mesh, gt_mesh_pv, 
                                              eval_path / f"invert_deltas/series_{eval_series_idx}_step_{idx}.gif")

            # 6. Recursive Reference Frame Update (The "Loop")
            if idx + 1 < series_end_idx:
                if canonical_frame:
                    x_rec_world = untransform_points(x_rec_np, rotations[idx], positions[idx])
                    x_rec_transformed = transform_points(x_rec_world, rotations[idx + 1], positions[idx + 1])
                else:
                    x_rec_transformed = x_rec_np  # already world frame -- no frame-hop needed
                x_recursive_jax = jnp.array(x_rec_transformed)[jnp.newaxis, ...]

            counter += 1

        # Final map to global dict using your requested clean loop
        for key in series_stats_dict:
            all_stats_dict[f'all_{key}'].append(series_stats_dict[key])
    
    plot_eval_series(all_stats_dict,
                      mode=plot_mode,
                      max_cols=max_cols,
                      n_step=n_step,
                      fill_variation=True,
                      add_mse=add_mse,
                      add_chamfer=add_chamfer,
                      add_hausdorff=add_hausdorff,
                      fig_path=eval_path/"eval_series.png")

    print(f"\nEvaluation of {num_series} series complete.")

    return all_stats_dict
    
def eval_time(config, trainer):
    import timeit
    import torch

    #NOTE: if data is randomly seeded the losses between iterations may be higher than expected
    data_path = config["datasets"]["data_out"]
    assert os.path.exists(data_path), "Dataset found"
    data  = np.load(data_path, allow_pickle=True)
    states = data['coords_t']
    action_features = config["network"]["action_features"]

    actions = actions_from_feature_map(action_features, data)
    x_sample = torch.tensor(states[0], dtype=torch.float32).T.unsqueeze(0).to(trainer.device)
    a_sample = torch.tensor(actions[0], dtype=torch.float32).unsqueeze(0).to(trainer.device)
    output_folder = Path(config["run"]["run_folder"])
    trainer.load(model_path = output_folder / "best_model.pth")
    trainer.net.eval()
    
    def forward(x, a):
        with torch.no_grad():
            return(trainer.net(x_t=x, a_t=a))
    def benchmark_step():
        if trainer.device.type == 'cuda':
            torch.cuda.synchronize()
        
        with torch.no_grad():
            forward(x_sample, a_sample)
        
        if trainer.device.type == 'cuda':
            torch.cuda.synchronize()

    runs = 10_000
    total_time = timeit.timeit(benchmark_step, number=runs)
    avg_time_ms = (total_time / runs) * 1000

    print(f"Forward pass avg time over {runs} runs: {avg_time_ms:.4f} ms")
    print(f"Total time for {runs} deformations: {total_time:.4f} ms")

def render_series(config, trainer, num_series=5, min_series_length=50, render_mode="point_cloud", max_workers=None, fps=10):
    '''
    Renders animations of the network's recursive predictions.
    
    :param render_mode: "point_cloud" (fast) or "mesh" (slow, uses invert_deltas_to_mesh)
    '''
    assert render_mode in ["point_cloud", "mesh"], "render_mode must be 'point_cloud' or 'mesh'"
    
    data_path = config["datasets"]["data_out"]
    data = np.load(data_path, allow_pickle=True)
    states = data['coords_t']
    states_tp1 = data['coords_tp1']
    action_features = config["network"]["action_features"]
    positions = data['positions']
    rotations = data['rotations']
    actions = actions_from_feature_map(action_features, data)
    series_lengths = data['series_lengths']
    
    output_folder = Path(config["run"]["run_folder"])
    render_path = output_folder / "renders"
    render_path.mkdir(exist_ok=True)
    
    def forward(x, a):
        with torch.no_grad():   
            return trainer.net(x_t=x, a_t=a)

    eval_series_idxs = [] 
    for idx, sl in enumerate(series_lengths):
        if sl >= min_series_length:
                eval_series_idxs.append(idx)
        if len(eval_series_idxs) >= num_series:
            break

    for eval_series_idx in tqdm(eval_series_idxs, desc=f"Rendering Series ({render_mode})"):
        series_length = series_lengths[eval_series_idx]
        series_start_idx = sum(series_lengths[:eval_series_idx])
        series_end_idx = series_start_idx + min(series_length, min_series_length) - 1
        
        pred_data = []
        gt_data = []
        previous_deltas = None
        
        x_recursive = torch.tensor(states[series_start_idx], dtype=torch.float32).T.unsqueeze(0).to(trainer.device)
        
        # mesh_data = data['meshes'][0]
        # base_faces = np.array(mesh_data['faces'])
        
        if render_mode == "mesh":
            base_mesh = pv.PolyData(mesh_data['points'], base_faces)

        for idx in tqdm(range(series_start_idx, series_end_idx), desc="Forward Pass", leave=False):
            
            a_t = torch.tensor(actions[idx]*5, dtype=torch.float32).unsqueeze(0).to(trainer.device)
            x_tp1_gt = torch.tensor(states_tp1[idx], dtype=torch.float32).T.unsqueeze(0).to(trainer.device)
            x_tp1_gt = x_tp1_gt.squeeze().cpu().numpy().T

            delta_recursive = forward(x_recursive, a_t)
            
            x_recursive = x_recursive + delta_recursive.transpose(1, 2) / 100
            x_rec_np = x_recursive.squeeze().cpu().numpy().T
            
            # gt_mesh_data = data['meshes_tp1'][idx]
            # gt_pts = np.array(gt_mesh_data['points'])

            # # if render_mode == "mesh":
            # #     tri_ids = data['tri_ids'][idx]
            # #     bary_coords = data['bary_coords'][idx]
            # #     recovered_mesh, current_deltas = invert_deltas_to_mesh(
            # #         base_mesh, x_rec_np, tri_ids, bary_coords, alpha=0.05, initial_guess=previous_deltas
            # #     )
            # #     previous_deltas = current_deltas
                
            # #     pred_data.append({'points': np.array(recovered_mesh.points), 'faces': np.array(recovered_mesh.faces)})
            # #     gt_data.append({'points': gt_pts, 'faces': base_faces})
            # # else:
            # #     # Point cloud mode: skip mesh reconstruction entirely!

            
            if idx + 1 < series_end_idx:
                x_rec_world = untransform_points(x_rec_np, rotations[idx], positions[idx])
                x_rec_transformed = transform_points(x_rec_world, rotations[idx + 1], positions[idx + 1])
                x_recursive = torch.tensor(x_rec_transformed, dtype=torch.float32).T.unsqueeze(0).to(trainer.device)
            
            pred_data.append({'points': x_rec_np, 'faces': None})
            gt_data.append({'points': x_tp1_gt, 'faces': None})

        gif_filename = f"series_{eval_series_idx}_{render_mode}.gif"
        out_filepath = str(render_path / gif_filename)
        
        create_deformation_gif_parallel(pred_data=pred_data, gt_data=gt_data, out_path=out_filepath, max_workers=max_workers, fps=fps)

if __name__ == "__main__":
    #Evaluate an existing trained model
    from forge_net.model.trainer import ForgeNetTrainer
    import yaml

    base_path = get_project_root()
    run_name = "jax_mse_2048_unmasked_unseeded"
    config_path = base_path / "runs" / run_name / "config_out.yml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    from main import make_dataloaders
    train_loader, test_loader = make_dataloaders(config)
    trainer = ForgeNetTrainer(config, train_loader, log_to_tb=False)
    output_folder = Path(config["run"]["run_folder"])
    eval_path = output_folder / "eval"
    eval_path.mkdir(exist_ok=True)

    trainer.load(model_path = output_folder / "checkpoints" / "248")
    # evaluate(config, trainer)
    evaluate_series(config, trainer,
                    add_mse=True,
                    add_chamfer=True, 
                    add_hausdorff=False,
                    plot_heatmaps=True,
                    plot_mode='dist',
                    num_series=1, min_series_length=60, 
                    n_step=5, max_cols=7, save_meshes=False)
    # # eval_time(config, trainer)

    # render_series(config, trainer, 
    #               num_series=1, 
    #               min_series_length=100,
    #               render_mode="point_cloud",
    #               max_workers=64, # Set to None to use all available cores
    #               fps=4)