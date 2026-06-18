from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from forge_net.data.dataloaders import GetSingleStepDataLoaders
from forge_net.utils.common import actions_from_feature_map
from forge_net.utils.common import MeshContainer, meshcontainer_to_pv
from forge_net.utils.math import *

def process_series(args):
    """Process a single series - this will run in parallel
    
    Args:
        args: Tuple of (series_id, group_df, total_points, press_width, mask_points, seed)
    """
    series_id, group_df, total_points, press_width, mask_points, seed = args
    series_coords_t = []
    series_coords_tp1 = []
    series_steps = []
    series_positions = []
    series_rotations = []
    pv_meshes = []
    pv_meshes_tp1 = []
    bary_coords_list = []
    tri_ids_list = []
    
    for i in range(len(group_df) - 1):
        row_t = group_df.iloc[i]
        row_tp1 = group_df.iloc[i + 1]

        # Input mesh (coords from frame i)
        mesh_data_t = json.loads(row_t["result"])
        num_steps_t = len(mesh_data_t["Steps"])
        vertices_t = np.array(mesh_data_t["Vertices"]).reshape(num_steps_t,-1)[-1]
        # vertices_t = mesh_data_t["Vertices"]
        triangles_t = mesh_data_t["Triangles"]
        tmp_mesh_t = MeshContainer.from_db(vertices_t, triangles_t)
        pv_mesh_t = meshcontainer_to_pv(tmp_mesh_t)

        mesh_data_tp1 = json.loads(row_tp1["result"])
        num_steps_tp1 = len(mesh_data_tp1["Steps"])
        vertices_tp1 = np.array(mesh_data_tp1["Vertices"]).reshape(num_steps_tp1,-1)[-1]
        # print(mesh_data_tp1, len(mesh_data_tp1))
        # print(mesh_data_tp1["Vertices"], len(mesh_data_tp1["Vertices"]))
        # vertices_tp1 = mesh_data_tp1["Vertices"]
        triangles_tp1 = mesh_data_tp1["Triangles"]
        tmp_mesh_tp1 = MeshContainer.from_db(vertices_tp1, triangles_tp1)
        pv_mesh_tp1 = meshcontainer_to_pv(tmp_mesh_tp1)
        
        s_tp1 = np.sum(np.array(mesh_data_tp1["Steps"]))
        p_tp1 = json.loads(row_tp1["position"])
        r_tp1 = json.loads(row_tp1["rotation"])
        
        pv_mesh_t.points = transform_points(np.array(pv_mesh_t.points), np.array(r_tp1), np.array(p_tp1))
        pv_mesh_tp1.points = transform_points(np.array(pv_mesh_tp1.points), np.array(r_tp1), np.array(p_tp1))
        
        if mask_points:

            tri_mask, _ = triangle_mask_from_window(
                    pv_mesh_t, center = 0.0, 
                    window_length=press_width, 
                    bc_length=3.0*press_width
                )
        else:

            tri_mask = None

        if total_points is not None:
            try:

                coords_t, point_triangle_ids, bary_coords = barycentric_sampling(
                    pv_mesh_t, total_points, tri_mask=tri_mask, seed=seed
                )
                
                tri_ids_list.append(point_triangle_ids)
                bary_coords_list.append(bary_coords)

                coords_tp1 = update_barycentric_points(pv_mesh_tp1, point_triangle_ids, bary_coords)

                
            except:
                print(f"Skipping hit in series {series_id} - no press contact")
                continue
        
        else:
            
            if tri_mask is not None:
                
                faces = np.array(pv_mesh_t.faces).reshape(-1, 4)[:, 1:]
                masked_vertex_indices = np.unique(faces[tri_mask].ravel())
                
                # Create arrays with all points
                coords_t = np.array(pv_mesh_t.points)
                coords_tp1 = np.array(pv_mesh_tp1.points)
                
                # Create a boolean mask for vertices (True for vertices to keep)
                vertex_mask = np.zeros(len(coords_t), dtype=bool)
                vertex_mask[masked_vertex_indices] = True
                
                # Zero out the non-masked vertices
                coords_t[~vertex_mask] = 0.0
                coords_tp1[~vertex_mask] = 0.0

            else:
                coords_t = pv_mesh_t.points
                coords_tp1 = pv_mesh_tp1.points
        
        series_coords_t.append(coords_t)
        series_coords_tp1.append(coords_tp1)
        series_steps.append(s_tp1)
        series_positions.append(p_tp1)
        series_rotations.append(r_tp1)
        #pv_meshes to enable pickling later with npz
        pv_meshes.append({
            'points': pv_mesh_t.points,
            'faces': pv_mesh_t.faces
        })
        pv_meshes_tp1.append({
            'points': pv_mesh_tp1.points,
            'faces': pv_mesh_tp1.faces
        })
        
    
    result = {
        'coords_t': series_coords_t,
        'coords_tp1': series_coords_tp1,
        'tri_ids': tri_ids_list,
        'bary_coords': bary_coords_list,
        'steps': series_steps,
        'positions': series_positions,
        'rotations': series_rotations,
        'length': len(series_coords_t),
        'series_id': series_id,
        'meshes': pv_meshes,
        'meshes_tp1' : pv_meshes_tp1
    }
    
    return result

def n_extract_data(db_path, total_points, lines, n_workers=None, mask_points=None, seed=None):
    """
    Extract data with parallel processing
    Args:
        db_path: Path to database
        total_points: Number of points to sample
        lines: Number of lines to read from DB
        n_workers: Number of parallel workers (None = use all CPUs)
        mask_points: If True use domain decomposition mask
        seed: if passed do deterministic sampling
    """
    # Read data from database
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM strike LIMIT {int(lines)};", conn)
    conn.close()
    
    # Prepare arguments for each series
    press_width = 1.0
    series_ids = df['series_id'].unique()
    args_list = [
        (series_id, 
         df[df['series_id'] == series_id].reset_index(drop=True), 
         total_points, 
         press_width,
         mask_points,
         seed
        )
        for series_id in series_ids
    ]
    
    # Process in parallel - need to pass compute flags
    if n_workers is None:
        n_workers = cpu_count()
    
    print(f"Processing {len(series_ids)} series using {n_workers} workers...")
    
    # Create partial function with compute flags
    from functools import partial
    process_func = partial(process_series)
    
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, args_list),
            total=len(args_list),
            desc="Processing series"
        ))
    
    # Initialize output dictionary with always-present keys
    output = {
        'coords_t': [],
        'coords_tp1': [],
        'tri_ids': [],
        'bary_coords': [],
        'steps': [],
        'positions': [],
        'rotations': [],
        'series_lengths': [],
        'series_ids': [],
        'meshes' : [],
        'meshes_tp1' : []
    }
    
    # Combine results
    for result in results:
        if result['length'] > 0:  # Only add series that produced data
            output['coords_t'].extend(result['coords_t'])
            output['coords_tp1'].extend(result['coords_tp1'])
            output['tri_ids'].extend(result['tri_ids'])
            output['bary_coords'].extend(result['bary_coords'])
            output['steps'].extend(result['steps'])
            output['positions'].extend(result['positions'])
            output['rotations'].extend(result['rotations'])
            output['series_lengths'].append(result['length'])
            output['series_ids'].append(result['series_id'])
            output['meshes'].extend(result['meshes'])
            output['meshes_tp1'].extend(result['meshes_tp1'])


    for key, value in output.items():
        if key in ['meshes', 'meshes_tp1']:
            arr = np.array(value, dtype=object)
        else:
            arr = np.array(value)
            if key in ['steps']:
                arr = arr.reshape(-1,1)
            
        output[key] = arr
    
    return output

def make_dataset(config):
    '''
    Processes a SQLite database into a numpy npz which is compatible with pytorch dataloaders
    '''
    total_points, mask_points, seed, data_out = config['datasets'].values()

    if Path(data_out).exists():
        print("Datasets already exists skipping creation")
        return
    
    db_path1, db_path2, lines = config['databases'].values()
    print(db_path1, db_path2)
    assert Path(db_path1).exists() and Path(db_path2).exists(), "Provided database paths do not exist check paths"
    
    data1 = n_extract_data(db_path1, total_points, lines, seed=seed, mask_points=mask_points, n_workers=128)
    data2 = n_extract_data(db_path2, total_points, lines, seed=seed,  mask_points=mask_points, n_workers=128)
    data = {key: np.concatenate((data1[key], data2[key]), axis=0) for key in data1.keys()}

    np.savez(data_out, **data)

def make_dataloaders(config):
    data_path = config["datasets"]["data_out"]
    data = np.load(data_path)
    c_t = data['coords_t']
    c_tp1 = data['coords_tp1']
    
    action_features = config["network"]["action_features"]
    # Build only what's in the config
    actions = actions_from_feature_map(action_features, data)
 
    train_loader, test_loader = GetSingleStepDataLoaders(
        coords_t=c_t,       
        coords_tp1=c_tp1,
        actions=actions,
        batch_size=config["network"]["batch_size"]
        )
    
    return(train_loader, test_loader)