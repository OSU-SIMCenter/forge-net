from pathlib import Path
import numpy as np
import yaml
from forge_net.data.dataloaders import * 
from forge_net.data.process_data import * 
from forge_net.utils.utils import *
from forge_net.model.trainer import Trainer
from forge_net.eval import *

def make_dataset(config):
    '''
    Processes a SQLite database into a numpy npz which is compatible with pytorch dataloaders
    '''
    ds_cfg = config['datasets']
    total_points = ds_cfg.get('points_per_state', ds_cfg.get('total_points'))
    mask_points = ds_cfg.get('mask_points')
    seed = ds_cfg.get('seed')
    data_out = ds_cfg.get('data_out')

    if data_out is not None and Path(str(data_out)).exists():
        print("Datasets already exists skipping creation")
        return
    
    db_cfg = config['databases']
    db_path1 = db_cfg.get('db1')
    db_path2 = db_cfg.get('db2')
    lines = db_cfg.get('lines')
    print(db_path1, db_path2)
    assert Path(str(db_path1)).exists() and Path(str(db_path2)).exists(), "Provided database paths do not exist check paths"
    
    data1 = n_extract_data(db_path1, total_points, lines, seed=seed, mask_points=mask_points, n_workers=128)
    data2 = n_extract_data(db_path2, total_points, lines, seed=seed,  mask_points=mask_points, n_workers=128)

    data = {}
    for key in data1.keys():
        if key in ['series_lengths', 'series_ids', 'meshes', 'meshes_tp1']:
            data[key] = data1[key] + data2[key]
        else:
            data[key] = np.vstack((data1[key], data2[key]))
    
    c_t = data['coords_t']
    c_tp1 = data['coords_tp1']
    s = data['steps']
    p = data['positions']
    r = data['rotations']

    series_lengths = data['series_lengths'] 
    series_ids = data['series_ids']
    np.savez(data_out, coords_t=c_t, coords_tp1=c_tp1, steps=s, positions=p, rotations=r, 
             series_lengths=np.array(series_lengths))
    
def make_dataloaders(config):
    data_path = config["datasets"]["data_out"]
    data = np.load(data_path)
    c_t = data['coords_t']
    c_tp1 = data['coords_tp1']
    
    action_features = config["network"]["action_features"]
    steps = data['steps']
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
 
    train_loader, test_loader = GetSingleStepDataLoaders(
        coords_t=c_t,       
        coords_tp1=c_tp1,
        actions=actions,
        batch_size=config["network"]["batch_size"]
        )
    
    return(train_loader, test_loader)

if __name__ == "__main__":

    base_path = get_project_root()
    config_path = base_path / "runs" / "mse_1024_unmaksed_seeded_w_tri_ids" / "config_out.yml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    run_folder = base_path / "runs" / config["run"]["run_name"]
    
    assert not run_folder.exists(), print("Run already exists")
    config["run"]["run_folder"] = run_folder

    train_loader, test_loader = make_dataloaders(config)
    trainer = Trainer(config, train_loader, test_loader)
    trainer.train()
    config = trainer.config # get any changes the trainer made to the config
    config["run"]["run_folder"] = str(run_folder) #yaml cannot dump Path Objects
    with open(run_folder / "config_out.yml", "w") as file:
        yaml.safe_dump(config, file)
    evaluate(config, trainer)
    evaluate_series(config, trainer)

    