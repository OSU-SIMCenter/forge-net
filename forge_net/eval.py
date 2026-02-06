from pathlib import Path
import torch
from forge_net.utils.utils import *
from forge_net.utils.plotting import * 
from pytorch3d.loss import chamfer_distance #original implementation uses a chamfer distance

def evaluate(config, trainer):
    
    data_path = config["datasets"]["data_out"]
    assert os.path.exists(data_path), "Dataset found"
    data  = np.load(data_path)
    states = data['coords_t']
    states_tp1 = data['coords_tp1']
    steps = data['steps']
    action_features = config["network"]["action_features"]
    positions = data['positions']
    rotations = data['rotations']

    # Define all possible features
    feature_map = {
        "steps": lambda: steps,
        "positions": lambda: positions[:, 0].reshape(-1, 1), #if positions we only care about translation in X
        "rotations": lambda: np.array([[quat_to_eulerxyz(quat)[0]] for quat in rotations]) #if rotations we only care about rotation about x
    }

    # Build only what's in the config
    actions = np.hstack([feature_map[f]() for f in action_features])
    
    output_folder = Path(config["run"]["run_folder"])
    eval_path = output_folder / "eval"
    eval_path.mkdir(exist_ok=True)


    trainer.load(model_path = output_folder / "best_model.pth")

    # plot_network_weights(trainer.state_dict, fig_path=eval_path / "network_hist.png") 
    
    def forward(x,a):
        with torch.no_grad():
            return(trainer.net(x_t=x,a_t=a))
    
    for idx in config["eval"]["eval_idxs"]:
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

        if config["network"]["loss"] == "mse":
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

def evaluate_series(config, trainer):
    
    data_path = config["datasets"]["data_out"]
    assert os.path.exists(data_path), "Dataset found"
    data  = np.load(data_path)
    states = data['coords_t']
    states_tp1 = data['coords_tp1']
    steps = data['steps']
    action_features = config["network"]["action_features"]
    positions = data['positions']
    rotations = data['rotations']
    series_lengths = data['series_lengths']

    # Define all possible features
    feature_map = {
        "steps": lambda: steps,
        "positions": lambda: positions[:, 0].reshape(-1, 1), #if positions we only care about translation in X
        "rotations": lambda: np.array([[quat_to_eulerxyz(quat)[0]] for quat in rotations]) #if rotations we only care about rotation about x
    }

    # Build only what's in the config
    actions = np.hstack([feature_map[f]() for f in action_features])
    output_folder = Path(config["run"]["run_folder"])
    eval_path = output_folder / "eval"
    eval_path.mkdir(exist_ok=True)


    trainer.load(model_path = output_folder / "best_model.pth")
    
    series_eval_idx = 0
    series_length = series_lengths[series_eval_idx]
    series_start_idx = sum(series_lengths[:series_eval_idx])
    series_end_idx = series_start_idx + series_lengths[series_eval_idx]
    # series_id = series_ids[series_eval_idx]
    print(f"Evaluating series {series_eval_idx} out of {len(series_lengths)} with \n \
            Series length: {series_length}  \
            Starting index: {series_start_idx} \
            End index: {series_end_idx} " )
    
    trainer.load(model_path=output_folder / "best_model.pth")
    trainer.net.eval() # Set to eval mode

    def forward(x, a):
        with torch.no_grad():
            return(trainer.net(x_t=x, a_t=a))

    gt_sequence = []
    single_step_preds = []
    recursive_preds = []
    losses = []
    dev_losses = []
    recursive_losses = []

    x_recursive = torch.tensor(states[series_start_idx], dtype=torch.float32).T.unsqueeze(0).to(trainer.device)
    
    current_r = rotations[series_start_idx + 1]
    current_p = positions[series_start_idx + 1]

    for idx in range(series_start_idx, series_end_idx):

        x_t_gt = torch.tensor(states[idx], dtype=torch.float32).T.unsqueeze(0).to(trainer.device)
        x_tp1_gt = torch.tensor(states_tp1[idx], dtype=torch.float32).T.unsqueeze(0).to(trainer.device)
        a_t = torch.tensor(actions[idx], dtype=torch.float32).unsqueeze(0).to(trainer.device)

        delta_hat = forward(x_t_gt, a_t)
        x_tp1_hat = x_t_gt + delta_hat.transpose(1, 2) / 100
        delta_rec = forward(x_recursive, a_t)
        x_recursive = x_recursive + delta_rec.transpose(1, 2) / 100
        
        step_loss = torch.mean((x_tp1_hat - x_tp1_gt)**2).item()*10_000
        dev_loss = torch.mean((x_recursive - x_tp1_hat)**2).item()
        rec_loss = torch.mean((x_recursive - x_tp1_gt)**2).item()


        if idx + 1 < series_end_idx:
            next_r = rotations[idx + 1]
            next_p = positions[idx + 1]
            
            # Convert to numpy, transform, convert back
            x_rec_np = x_recursive.squeeze().cpu().numpy().T
            x_rec_world = untransform_points(x_rec_np, current_r, current_p)
            x_rec_transformed = transform_points(x_rec_world, next_r, next_p)
            x_recursive = torch.tensor(x_rec_transformed, dtype=torch.float32).T.unsqueeze(0).to(trainer.device)

            current_r = next_r
            current_p = next_p
    
        
        gt_sequence.append(x_t_gt.squeeze().cpu().numpy().T)
        single_step_preds.append(x_tp1_hat.squeeze().cpu().numpy().T)
        recursive_preds.append(x_recursive.squeeze().cpu().numpy().T)
        losses.append(step_loss)
        dev_losses.append(dev_loss)
        recursive_losses.append(rec_loss)
    losses[0] = losses[0] / 10_000
    plot_eval_series(gt_sequence, single_step_preds, recursive_preds, 
                        losses, dev_losses, recursive_losses, n_step=1, max_cols=20,
                        fig_path=eval_path/"eval_series.png")

if __name__ == "__main__":
    #Evaluate an existing trained model
    from model.trainer import Trainer
    from utils.utils import * 
    import yaml

    base_path = get_project_root()
    run_name = "mse_1024_unmasked"
    config_path = base_path / "runs" / run_name / "config_out.yml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    from main import make_dataloaders
    train_loader, test_loader = make_dataloaders(config)
    trainer = Trainer(config, train_loader, log_to_tb=False)
    # evaluate(config, trainer)
    evaluate_series(config, trainer)