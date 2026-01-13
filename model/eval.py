from pathlib import Path
import torch
from utils.plotting import * 

def evaluate(config, trainer):
    
    data_path = config["datasets"]["data_out"]
    assert os.path.exists(data_path), "Dataset found"
    data  = np.load(data_path)
    states = data['coords_t']
    states_tp1 = data['coords_tp1']
    steps = data['steps']
    action_features = config["network"]["action_features"]
    steps = data['steps']
    positions = data['positions']
    rotations = data['rotations']

    # Define all possible features
    feature_map = {
        "steps": lambda: steps,
        "positions": lambda: positions[:, 0].reshape(-1, 1),
        "rotations": lambda: rotations
    }
    actions = np.hstack([feature_map[f]() for f in action_features])
    
    output_folder = config["run"]["run_folder"]
    eval_path = Path(output_folder + "eval/").mkdir()

    trainer.load(model_path = output_folder / "model.pth")
    plot_network_weights(trainer.state_dict) 

    
    for idx in config["eval"]["eval_idxs"]:

        x_t = torch.tensor(states[idx], dtype=torch.float32).T.unsqueeze(0) # x.shape = 1,3,n_points
        x_tp1 = torch.tensor(states_tp1[idx], dtype=torch.float32).T.unsqueeze(0) # x.shape = 1,3,n_points

        a = torch.tensor(actions[idx], dtype=torch.float32).unsqueeze(0) #a.shape = 1,8
        x_t_np = x_t.T.squeeze().numpy()
        x_tp1_np = x_tp1.T.squeeze().numpy()
        detla_gt = x_tp1 - x_t

        delta_hat = trainer.net(x_t=x_t, a_t=a)
        x_tp1_hat = x_t.squeeze(0).T + delta_hat.squeeze(0)
        x_tp1_hat = x_tp1_hat.numpy()
        delta_hat = delta_hat.squeeze(0).T
        delta_gt = x_tp1 - x_t

        loss_cont = torch.sum(((delta_hat - delta_gt) ** 2),axis=1).numpy()
        print(loss_cont)
        #Make eval plots
        compare_scatters(pc1=x_tp1_np, pc2=x_tp1_hat, 
                    label_1="Ground Truth Mesh Tp1", 
                    label_2="Predicted Mesh Tp1", 
                    label_3='G.T. vs. Predicted')

        visualize_point_diff(x_tp1_hat,x_tp1_np, point_size=5, 
                             label="Vector Field Error \n(Predicted Deltas minus G.T. Deltas)",
                              fig_path = eval_path + "point_diff.png")
        
        compare_vector_fields(x_t_np, x_tp1_np, x_tp1_hat, min_magnitude=8.0, 
                              fig_path=eval_path + "compare_vector_fields.png")
        
        compare_scatters_w_loss_cont(pc1=x_t_np, pc2=x_tp1_hat, loss_cont=loss_cont, 
                                     fig_path=eval_path + "compare_scatter_loss.png")
        
        visualize_vector_diff_w_loss_cont(x_t=x_t_np, x_tp1=x_tp1_np, x_hat=x_tp1_hat, 
                                            loss_cont=loss_cont[-1], min_magnitude=8.0,
                                            fig_path = eval_path + "vector_fields_loss.png")

if __name__ == "__main__":
    #Evaluate an existing trained model
    from model.trainer import Trainer
    train_loader, test_loader = make_dataloaders(config)
    trainer = Trainer(config, train_loader, test_loader)
    evaluate(trainer, trainer)