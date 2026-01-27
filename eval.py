from pathlib import Path
import torch
from utils.utils import *
from utils.plotting import * 
from pytorch3d.loss import chamfer_distance #original implementation uses a chamfer distance
from loss.loss import AdaptiveSlicedWasserstein

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
        
        # elif config["network"]["loss"] == "wsd": #issue with loss_cont for each point
        #     loss_cont = AdaptiveSlicedWasserstein(device=torch.device("cpu"))\
        #                                             (delta_hat.T.unsqueeze(0), 
        #                                              delta_gt.T.unsqueeze(0))
        
        else:
            assert "Loss function not found"
            
        #Make eval plots
        compare_scatters(pc1=x_tp1_np, pc2=x_tp1_hat, 
                    label_1="Ground Truth Mesh Tp1", 
                    label_2="Predicted Mesh Tp1", 
                    label_3='G.T. vs. Predicted',
                    fig_path= idx_path / f"compare_scatters_{idx}.png")

        visualize_point_diff(x_tp1_hat,x_tp1_np, point_size=5, 
                             label="Vector Field Error \n(Predicted Deltas minus G.T. Deltas)",
                              fig_path = idx_path / f"point_diff_{idx}.png")
        
        compare_vector_fields(x_t_np, x_tp1_np, x_tp1_hat, min_magnitude=8.0, 
                              fig_path=idx_path / f"compare_vector_fields_{idx}.png")
        
        compare_scatters_w_loss_cont(pc1=x_t_np, pc2=x_tp1_hat, loss_cont=loss_cont, 
                                     fig_path=idx_path / f"compare_scatter_loss_{idx}.png")
        #plot corresponding
        visualize_vector_diff_w_loss_cont(x_t=x_t_np, x_tp1=x_tp1_np, x_hat=x_tp1_hat, 
                                            loss_cont=loss_cont, min_magnitude=8.0,
                                            fig_path = idx_path / f"vector_fields_loss_{idx}.png")
        #plot nearest
        visualize_vector_diff_w_loss_cont(x_t=x_t_np, x_tp1=x_tp1_np, x_hat=x_tp1_hat, 
                                            loss_cont=loss_cont, min_magnitude=8.0,
                                            use_nearest=True,
                                            fig_path = idx_path / f"vector_nn_fields_loss_{idx}.png")
if __name__ == "__main__":
    #Evaluate an existing trained model
    from model.trainer import Trainer
    from utils.utils import * 
    import yaml

    base_path = get_project_root()
    run_name = "mse_256tmsamples_seeded"
    config_path = base_path / "runs" / run_name / "config_out.yml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    from main import make_dataloaders
    train_loader, test_loader = make_dataloaders(config)
    trainer = Trainer(config, train_loader)
    evaluate(config, trainer)

    