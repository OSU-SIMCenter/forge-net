from pathlib import Path
import torch
from forge_net.utils.common import *
from forge_net.utils.math import *
from forge_net.utils.plotting import * 
from forge_net.utils.invert_deltas import invert_deltas_to_mesh, save_comparison_turntable
from forge_net.loss.chamfer import chamfer_distance
# from pytorch3d.loss import chamfer_distance #original implementation uses a chamfer distance
from tqdm import tqdm 

def evaluate(config, trainer):
    '''
    Evaluate a trained ForgeNet network by creating some simple scatter and vector plots
    
    :param config: YAML config containing paths to data and network settings
    :param trainer: trainer class instance
    '''
    data_path = config["datasets"]["data_out"]
    assert os.path.exists(data_path), "Dataset found"
    data  = np.load(data_path)
    states = data['coords_t']
    states_tp1 = data['coords_tp1']
    action_features = config["network"]["action_features"]
    
    actions = actions_from_feature_map(action_features, data)
    
    output_folder = Path(config["run"]["run_folder"])
    eval_path = output_folder / "eval"
    eval_path.mkdir(exist_ok=True)

    trainer.load(model_path = output_folder / "best_model.pth")
    
    def forward(x,a):
        with torch.no_grad():
            return(trainer.net(x_t=x,a_t=a))
    
    for idx in config["eval"]["eval_idxs"]:
        idx = 0
        idx_path = eval_path / str(idx)
        idx_path.mkdir(exist_ok=True)

        x_t = torch.tensor(states[idx], dtype=torch.float32).T.unsqueeze(0).to(trainer.device) # x.shape = 1,3,n_points   
        x_tp1 = torch.tensor(states_tp1[idx], dtype=torch.float32).T.unsqueeze(0).to(trainer.device) # x.shape = 1,3,n_points

        a = torch.tensor(actions[idx], dtype=torch.float32).unsqueeze(0).to(trainer.device) #a.shape = 1,8
        x_t_np = x_t.T.squeeze().cpu().numpy()
        x_tp1_np = x_tp1.T.squeeze().cpu().numpy()
        delta_hat = forward(x_t,a) / 100
        x_tp1_hat = x_t.squeeze(0).T + delta_hat.squeeze(0)
        x_tp1_hat = x_tp1_hat.cpu().numpy()
        delta_hat = delta_hat.cpu().squeeze().T
        delta_gt = (x_tp1 - x_t).squeeze().cpu()    

        if config["network"]["loss"] == "mse": #TODO - these should be x_t and x_tp1
            loss_cont = torch.sum(((delta_hat - delta_gt) ** 2),axis=0).squeeze(0).numpy()
        
        elif config["network"]["loss"] == "chamfer" or config["network"]["loss"] == "wsd":
            loss_cont =  chamfer_distance(delta_gt.T.unsqueeze(0), 
                                          delta_hat.T.unsqueeze(0), 
                                          point_reduction=None, 
                                          batch_reduction=None)[0][0][-1]
        
        else:
            raise ValueError("Loss function not found")

        #Make eval plots
        compare_scatters(pc1=x_tp1_np, pc2=x_tp1_hat, 
                    label_1="Ground Truth Mesh Tp1", 
                    label_2="Predicted Mesh Tp1", 
                    label_3='G.T. vs. Predicted',
                    fig_path= idx_path / f"compare_scatters_{idx}.png")
        
        compare_scatters_w_loss_cont(pc1=x_t_np, pc2=x_tp1_hat, loss_cont=loss_cont, 
                                     fig_path=idx_path / f"compare_scatter_loss_{idx}.png")

        visualize_point_diff(x_tp1_hat,x_tp1_np, point_size=5, 
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

def evaluate_series(config, trainer, num_series, min_series_length, 
                    max_cols, plot_mode, n_step, add_mse, add_chamfer, add_hausdorff, save_meshes):
    '''
    Evaluate a trained ForgeNet network over multiple series of hits recursively
    
    :param config: YAML config containing paths to data and network settings
    :param trainer: trainer class instance
    :param num_series: How many series to evaluate total (e.g n=25)
    :param min_series_length: Minimum rollout length
    :param max_cols: maximum columns for the series scatters (can be very wide)
    :param plot_mode: "dist" or "loss" - which metrics to plot
    :param n_step: Make plots ever n_steps
    :param add_chamfer: whether to calculate and add chamfer distance to the eval plot
    :param add_hausdorff: Uses LSQR to reconstruct surface mesh and compute hausdorff distance
    :param save_meshes: Whether to save reconstructed mesh (or simply use it for a metric )
    '''
    #NOTE: if data is randomly seeded the losses between iterations may be higher than expected
    data_path = config["datasets"]["data_out"]
    assert os.path.exists(data_path), "Dataset found"
    data  = np.load(data_path, allow_pickle=True)
    states = data['coords_t']
    states_tp1 = data['coords_tp1']
    steps = data['steps']
    action_features = config["network"]["action_features"]
    positions = data['positions']
    rotations = data['rotations']

    actions = actions_from_feature_map(action_features, data)
    
    def forward(x, a):
        with torch.no_grad():
            return(trainer.net(x_t=x, a_t=a))
    
    series_lengths = data['series_lengths']
    series_ids = data['series_ids']
    eval_series_idxs = [] 
    for idx, sl in enumerate(series_lengths):
        if sl >= min_series_length:
                eval_series_idxs.append(idx)
        if len(eval_series_idxs) >= num_series:
            break
    
    all_stats_dict = {
        'all_gt_steps': [],
        'all_one_step_preds': [],
        'all_rec_step_preds': [],
        'all_one_step_dist_means' : [],
        'all_one_step_dist_stds' : [],
        'all_one_step_dist_95pct_means' : [],
        'all_one_step_dist_95pct_stds' : [],
        'all_one_step_mses' : [],
        'all_rec_step_dist_means' : [],
        'all_rec_step_dist_stds' : [],
        'all_rec_step_dist_95pct_means' : [],
        'all_rec_step_dist_95pct_stds' : [],
        'all_rec_step_mses' : [],
        'all_rec_step_chamfers' : [],
        'all_rec_step_hausdorffs' : [] } 

    for eval_series_idx in tqdm(eval_series_idxs):
        series_length = series_lengths[eval_series_idx]
        series_start_idx = sum(series_lengths[:eval_series_idx])
        series_end_idx = series_start_idx + series_length - 1 if series_length <= min_series_length else series_start_idx + min_series_length - 1
        truncated_length = series_end_idx - series_start_idx
        last_frame_idx = series_end_idx - 1

        print(f"Evaluating series {eval_series_idx} out of {len(series_lengths)} with \n \
                Series length: {series_length}  \
                Starting index: {series_start_idx} \
                End index: {series_end_idx} \
                Truncated length: {truncated_length}" )

        
        counter = 0
        previous_deltas = None

        series_stats_dict = {
            'gt_steps': [],
            'one_step_preds': [],
            'rec_step_preds': [],
            'one_step_dist_means': [],
            'one_step_dist_stds': [],
            'one_step_dist_95pct_means': [],
            'one_step_dist_95pct_stds': [],
            'one_step_mses': [],
            'rec_step_dist_means': [],
            'rec_step_dist_stds': [],
            'rec_step_dist_95pct_means': [],
            'rec_step_dist_95pct_stds': [],
            'rec_step_mses': [],
            'rec_step_chamfers': [],
            'rec_step_hausdorffs': [],
            'rec_step_chamfer_to_last': [],
            'rec_step_mse_to_last': []  }

        x_recursive = torch.tensor(states[series_start_idx], dtype=torch.float32).T.unsqueeze(0).to(trainer.device)

        for idx in tqdm(range(series_start_idx, series_end_idx)):
            x_t_gt = torch.tensor(states[idx], dtype=torch.float32).T.unsqueeze(0).to(trainer.device)
            x_tp1_gt = torch.tensor(states_tp1[idx], dtype=torch.float32).T.unsqueeze(0).to(trainer.device)
            a_t = torch.tensor(actions[idx], dtype=torch.float32).unsqueeze(0).to(trainer.device)
            delta_hat = forward(x_t_gt, a_t)
            x_tp1_hat = x_t_gt + delta_hat.transpose(1, 2) / 100
            delta_recursive = forward(x_recursive, a_t)
            
            if idx != series_start_idx:
                x_recursive = x_recursive + delta_recursive.transpose(1, 2) / 100
                 
            one_step_sq_diff_arr = ((x_tp1_hat - x_tp1_gt)**2).squeeze(0).cpu().numpy()
            rec_step_sq_diff_arr = ((x_recursive - x_tp1_gt)**2).squeeze(0).cpu().numpy()

            one_step_dist_arr = (np.sum(one_step_sq_diff_arr, axis=0))**0.5 # gives 1 distance per point (x,y,z) -> d
            rec_step_dist_arr = (np.sum(rec_step_sq_diff_arr, axis=0))**0.5
            
            #compute one_step statistics
            one_step_mse = np.mean(one_step_sq_diff_arr)            
            one_step_dist_mean = np.mean(one_step_dist_arr)
            one_step_dist_std = np.std(one_step_dist_arr)
            one_step_dist_95pct = np.percentile(one_step_dist_arr, 95)
            one_step_dist_95pct_arr = np.array([x for x in one_step_dist_arr if x >= one_step_dist_95pct]) #can also do this with np.extract?
            one_step_dist_95pct_mean = np.mean(one_step_dist_95pct_arr)
            one_step_dist_95pct_std = np.std(one_step_dist_95pct_arr)

            #compute rec_step statistics
            rec_step_mse = np.mean(rec_step_sq_diff_arr)
            rec_step_dist_mean = np.mean(rec_step_dist_arr)
            rec_step_dist_std = np.std(rec_step_dist_arr)
            rec_step_95pct = np.percentile(rec_step_dist_arr, 95)
            rec_step_step_95pct_arr = np.array([x for x in rec_step_dist_arr if x >= rec_step_95pct])
            rec_step_95pct_mean = np.mean(rec_step_step_95pct_arr)
            rec_step_95pct_std = np.std(rec_step_step_95pct_arr)

            #save everything
            series_stats_dict['gt_steps'].append(x_t_gt.squeeze().cpu().numpy().T)
            series_stats_dict['one_step_preds'].append(x_tp1_hat.squeeze().cpu().numpy().T)
            series_stats_dict['rec_step_preds'].append(x_recursive.squeeze().cpu().numpy().T)
            series_stats_dict['one_step_dist_means'].append(one_step_dist_mean)
            series_stats_dict['one_step_dist_stds'].append(one_step_dist_std)
            series_stats_dict['one_step_dist_95pct_means'].append(one_step_dist_95pct_mean)
            series_stats_dict['one_step_dist_95pct_stds'].append(one_step_dist_95pct_std)
            series_stats_dict['one_step_mses'].append(one_step_mse)
            series_stats_dict['rec_step_dist_means'].append(rec_step_dist_mean)
            series_stats_dict['rec_step_dist_stds'].append(rec_step_dist_std)
            series_stats_dict['rec_step_dist_95pct_means'].append(rec_step_95pct_mean)
            series_stats_dict['rec_step_dist_95pct_stds'].append(rec_step_95pct_std)
            series_stats_dict['rec_step_mses'].append(rec_step_mse)

            if add_chamfer:
                rec_step_chamfer =  chamfer_distance(x_recursive.permute(0,2,1), x_tp1_gt.permute(0,2,1))[0].item()
                series_stats_dict['rec_step_chamfers'].append(rec_step_chamfer)

                #calcualte chamfer from current frame to goal (last) frame
                # Transform x_recursive into the last frame's reference frame
                x_rec_np = x_recursive.squeeze().cpu().numpy().T
                x_rec_world = untransform_points(x_rec_np, rotations[idx], positions[idx])
                x_rec_in_last_frame = transform_points(x_rec_world, rotations[last_frame_idx], positions[last_frame_idx])
                
                # Get the last frame gt points (preloaded before loop to avoid slow pickle access)
                x_last_gt = torch.tensor(states_tp1[last_frame_idx], dtype=torch.float32).unsqueeze(0).to(trainer.device)
                x_rec_last_frame_tensor = torch.tensor(x_rec_in_last_frame, dtype=torch.float32).unsqueeze(0).to(trainer.device)
                
                chamfer_to_last = chamfer_distance(
                    x_rec_last_frame_tensor,
                    x_last_gt
                )[0].item()
                print(chamfer_to_last)
                series_stats_dict['rec_step_chamfer_to_last'].append(chamfer_to_last)



            if idx + 1 < series_end_idx:
                #Transform into next frame reference
                x_rec_np = x_recursive.squeeze().cpu().numpy().T
                x_rec_world = untransform_points(x_rec_np, rotations[idx], positions[idx])
                x_rec_transformed = transform_points(x_rec_world, rotations[idx + 1], positions[idx + 1])
                x_recursive = torch.tensor(x_rec_transformed, dtype=torch.float32).T.unsqueeze(0).to(trainer.device)


            if add_hausdorff: # mesh reconstruction is pretty slow
                mesh_data = data['meshes'][0]
                base_mesh = pv.PolyData(mesh_data['points'], mesh_data['faces'])
                tri_ids = data['tri_ids'][idx]
                bary_coords = data['bary_coords'][idx]
                gt_mesh = data['meshes_tp1'][idx]
            
                recovered_mesh, current_deltas = invert_deltas_to_mesh(
                                                                        base_mesh, 
                                                                        x_rec_np, 
                                                                        tri_ids, 
                                                                        bary_coords, 
                                                                        alpha=0.05,
                                                                        initial_guess=previous_deltas
                                                                    )
                previous_deltas = current_deltas
                gt_mesh_pv = pv.PolyData(gt_mesh['points'], mesh_data['faces'])
                filename = f"comparison_step_{idx}.gif"
                counter += 1
                if save_meshes and counter % n_step == 0:
                    # print("Saving recon'd mesh turntables")
                    save_comparison_turntable(recovered_mesh, 
                                            gt_mesh_pv, 
                                            eval_path /"surfaces/invert_deltas"/filename, 
                                            n_frames=300, 
                                            fps=15)
                #compute hausdorff distance
                rec_step_hausdorff = compute_haussdorff_distance(pv_mesh1=gt_mesh_pv, pv_mesh2=recovered_mesh)
                series_stats_dict['rec_step_hausdorffs'].append(rec_step_hausdorff)
                
            
        #save everything again
        all_stats_dict['all_gt_steps'].append(series_stats_dict['gt_steps'])
        all_stats_dict['all_one_step_preds'].append(series_stats_dict['one_step_preds'])
        all_stats_dict['all_rec_step_preds'].append(series_stats_dict['rec_step_preds'])
        all_stats_dict['all_one_step_dist_means'].append(series_stats_dict['one_step_dist_means'])
        all_stats_dict['all_one_step_dist_stds'].append(series_stats_dict['one_step_dist_stds'])
        all_stats_dict['all_one_step_dist_95pct_means'].append(series_stats_dict['one_step_dist_95pct_means'])
        all_stats_dict['all_one_step_dist_95pct_stds'].append(series_stats_dict['one_step_dist_95pct_stds'])
        all_stats_dict['all_one_step_mses'].append(series_stats_dict['one_step_mses'])
        all_stats_dict['all_rec_step_dist_means'].append(series_stats_dict['rec_step_dist_means'])
        all_stats_dict['all_rec_step_dist_stds'].append(series_stats_dict['rec_step_dist_stds'])
        all_stats_dict['all_rec_step_dist_95pct_means'].append(series_stats_dict['rec_step_dist_95pct_means'])
        all_stats_dict['all_rec_step_dist_95pct_stds'].append(series_stats_dict['rec_step_dist_95pct_stds'])
        all_stats_dict['all_rec_step_mses'].append(series_stats_dict['rec_step_mses'])
        all_stats_dict['all_rec_step_chamfers'].append(series_stats_dict['rec_step_chamfers'])
        all_stats_dict['all_rec_step_hausdorffs'].append(series_stats_dict['rec_step_hausdorffs'])

    
    plot_eval_series(all_stats_dict,
                      mode=plot_mode,
                      max_cols=max_cols,
                      n_step=n_step,
                      fill_variation=True,
                      add_mse=add_mse,
                      add_chamfer=add_chamfer,
                      add_hausdorff=add_hausdorff,
                      fig_path=eval_path/"eval_series.png")
                      
    print(f"Saving output to {eval_path}")

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
        
        
        if render_mode == "mesh":
            base_mesh = pv.PolyData(mesh_data['points'], base_faces)

        for idx in tqdm(range(series_start_idx, series_end_idx), desc="Forward Pass", leave=False):
            
            a_t = torch.tensor(actions[idx]*5, dtype=torch.float32).unsqueeze(0).to(trainer.device)
            x_tp1_gt = torch.tensor(states_tp1[idx], dtype=torch.float32).T.unsqueeze(0).to(trainer.device)
            x_tp1_gt = x_tp1_gt.squeeze().cpu().numpy().T

            delta_recursive = forward(x_recursive, a_t)
            
            x_recursive = x_recursive + delta_recursive.transpose(1, 2) / 100
            x_rec_np = x_recursive.squeeze().cpu().numpy().T
            
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
    run_name = "mse_1024_unmasked_seeded_tri_ids"
    config_path = base_path / "runs" / run_name / "config_out.yml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    from main import make_dataloaders
    train_loader, test_loader = make_dataloaders(config)
    trainer = ForgeNetTrainer(config, train_loader, log_to_tb=False)
    output_folder = Path(config["run"]["run_folder"])
    eval_path = output_folder / "eval"
    eval_path.mkdir(exist_ok=True)

    trainer.load(model_path = output_folder / "best_model.pth")
    trainer.net.eval()
    evaluate(config, trainer)
    evaluate_series(config, trainer,
                    add_mse=False,
                    add_chamfer=False, 
                    add_hausdorff=False,
                    plot_mode='dist',
                    num_series=1, min_series_length=60, 
                    n_step=1, max_cols=7, save_meshes=True)
    # eval_time(config, trainer)

    # render_series(config, trainer, 
    #               num_series=1, 
    #               min_series_length=100,
    #               render_mode="point_cloud",
    #               max_workers=64, # Set to None to use all available cores
    #               fps=4)