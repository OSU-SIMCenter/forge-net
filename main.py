import numpy as np
import pyvista as pv
from process_data import extract_data
pv.start_xvfb()
pv.set_jupyter_backend('static')
from scipy.spatial.transform import Rotation

import numpy as np
import torch
import torch.optim as optim
from dataloaders import GetSingleStepDataLoaders
import model
from train import train_model


total_points = 2048
lines = 10_000
db_path1 = '/local/scratch/groves/jax-forgeRL/jax-forge/data/tianhong_data/noisy_cogging.db'
db_path2 = '/local/scratch/groves/jax-forgeRL/jax-forge/data/tianhong_data/RL_random_cogging.db'
c_t1, c_tp11, s1, p1, r1, series_lengths1, series_ids1 = extract_data(db_path1, total_points, lines)
c_t2, c_tp12, s2, p2, r2, series_lengths2, series_ids2 = extract_data(db_path2, total_points, lines)

c_t = np.vstack((c_t1, c_t2))
c_tp1 = np.vstack((c_tp11, c_tp12))
s = np.vstack((s1, s2))
p = np.vstack((p1, p2))
r = np.vstack((r1,r2))
series_lengths = series_lengths1 + series_lengths2
series_ids = series_ids1 + series_ids2

# group_df = extract_data(db_path, total_points, lines)

# group_df.head()

np.savez('./test_comb_FOR.npz', coords_t=c_t, coords_tp1=c_tp1, steps=s, positions=p, rotations=r, series_lengths=np.array(series_lengths))


data  = np.load('/local/scratch/groves/jax-forgeRL/models/forging_autoencoder/data/test_comb_FOR.npz')
c_t = data['coords_t']
c_tp1 = data['coords_tp1']
steps = data['steps']
positions = data['positions']
rotations = data['rotations']
# actions = np.hstack((steps, positions, rotations))
actions = steps
point_size = c_t.shape[1]
batch_size = 32
output_folder = "./output_cogging/"
save_results = True
use_GPU = True
latent_size = 256
epochs = 200
train_loader, test_loader = GetSingleStepDataLoaders(
    coords_t=c_t,       
    coords_tp1=c_tp1,
    actions=actions,
    batch_size=batch_size
)

net = model.PCTransitionModel(point_size, latent_size)
batch_size = 32
output_folder = "./output_cogging/"
save_results = True
use_GPU = True
latent_size = 256
epochs = 200

if(use_GPU):
    device = torch.device("cuda:0")
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
else:
    device = torch.device("cpu")

net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.0005)

train_model(train_loader, test_loader, net, epochs, optimizer, device, save_results, output_folder)