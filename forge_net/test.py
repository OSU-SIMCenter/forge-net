
import numpy as np
from forge_net.utils.utils import get_project_root
import yaml

base_path = get_project_root()
run_name = 'mse_1024_masked_nores'
config_path = base_path / 'runs' / run_name / 'config_out.yml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

data_path = config['datasets']['data_out']
data = np.load(data_path)
states = data['coords_t']
states_tp1 = data['coords_tp1']
steps = data['steps']
idx = 4
x_t_0 = states[idx]      # Mesh at t=0, in frame 0
x_tp1_0 = states_tp1[idx] # Deformed mesh at t=1, also in frame 0

print('Sample 0:')
print(f'  x_t[0] x-range: [{x_t_0[:,0].min():.4f}, {x_t_0[:,0].max():.4f}], span: {x_t_0[:,0].max() - x_t_0[:,0].min():.4f}')
print(f'  x_tp1[0] x-range: [{x_tp1_0[:,0].min():.4f}, {x_tp1_0[:,0].max():.4f}], span: {x_tp1_0[:,0].max() - x_tp1_0[:,0].min():.4f}')
print(f'  Delta span in x: {(x_tp1_0[:,0].max() - x_tp1_0[:,0].min()) - (x_t_0[:,0].max() - x_t_0[:,0].min()):.4f}')
print(f'  Steps: {steps[idx]}')

# Check if they're the same points (just deformed)
# If same points, the y and z coordinates should be similar (less affected by forging)
print(f'\\n  y-range x_t[0]: [{x_t_0[:,1].min():.4f}, {x_t_0[:,1].max():.4f}]')
print(f'  y-range x_tp1[0]: [{x_tp1_0[:,1].min():.4f}, {x_tp1_0[:,1].max():.4f}]')

# Check point correspondence - are these the same sampled points?
# If same barycentric sampling, the mean should shift uniformly
print(f'\\n  Mean position x_t[0]: [{x_t_0[:,0].mean():.4f}, {x_t_0[:,1].mean():.4f}, {x_t_0[:,2].mean():.4f}]')
print(f'  Mean position x_tp1[0]: [{x_tp1_0[:,0].mean():.4f}, {x_tp1_0[:,1].mean():.4f}, {x_tp1_0[:,2].mean():.4f}]')
print(f'  Mean delta: [{(x_tp1_0[:,0] - x_t_0[:,0]).mean():.4f}, {(x_tp1_0[:,1] - x_t_0[:,1]).mean():.4f}, {(x_tp1_0[:,2] - x_t_0[:,2]).mean():.4f}]')

