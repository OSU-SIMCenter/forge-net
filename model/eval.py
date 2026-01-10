import os 
import torch
def evaluate(config, trainer):
    
    output_folder = config["run"]["output_folder"]
    trainer.load(model_path = output_folder  + "model.pth")
    eval_path = os.mkdir(output_folder+"eval/")

    x_t = torch.tensor(states[idx], dtype=torch.float32).T.unsqueeze(0) # x.shape = 1,3,n_points
    x_tp1 = torch.tensor(states_tp1[idx], dtype=torch.float32).T.unsqueeze(0) # x.shape = 1,3,n_points

    a = torch.tensor(actions[idx], dtype=torch.float32).unsqueeze(0) #a.shape = 1,8
    x_t_np = x_t.T.squeeze().numpy()
    x_tp1_np = x_tp1.T.squeeze().numpy()
    detla_gt = x_tp1 - x_t