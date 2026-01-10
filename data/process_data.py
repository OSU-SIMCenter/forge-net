import sqlite3
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils.utils import *

def process_single_series(args, compute_bc_mask=False, compute_spatial_features=False):
    """Process a single series - this will run in parallel
    
    Args:
        args: Tuple of (series_id, group_df, total_points, press_width)
        compute_bc_mask: If True, compute binary mask for boundary condition points
        compute_spatial_features: If True, compute distance to BC edge and direction to BC center
    """
    series_id, group_df, total_points, press_width = args
    series_points_t = []
    series_points_tp1 = []
    series_steps = []
    series_positions = []
    series_rotations = []
    pv_meshes = []
    pv_meshes_tp1 = []
    
    # Optional: boundary condition features
    series_bc_masks = [] if compute_bc_mask else None
    series_distances_to_bc_edge = [] if compute_spatial_features else None
    series_contact_directions = [] if compute_spatial_features else None
    
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


        try:
            tri_mask, _ = triangle_mask_from_window(
                pv_mesh_t, center = 0.0, 
                window_length=press_width, 
                bc_length=3.0*press_width
            )
            coords_t, point_triangle_ids, bary_coords = barycentric_sampling(
                pv_mesh_t, total_points, tri_mask=None
            )
            
            coords_tp1 = update_barycentric_points(pv_mesh_tp1, point_triangle_ids, bary_coords)
            
        except:
            print(f"Skipping hit in series {series_id} - no press contact")
            continue
        
                
        # Compute BC features using existing triangle_mask_from_window
        if compute_bc_mask or compute_spatial_features:
            # Get BC boundaries (bc_length=0 means just the press window)
            _, bc_bounds = triangle_mask_from_window(
                pv_mesh_t, 
                center = 0.0,  
                window_length=press_width,
                bc_length=0.0  #
            )
            # Binary mask: 1 if point is inside BC, 0 otherwise
            point_x = coords_t[:, 0]
            bc_lower = bc_bounds[0]
            bc_upper = bc_bounds[1]
            bc_mask = ((point_x >= bc_lower) & (point_x <= bc_upper)).astype(np.float32)
            series_bc_masks.append(bc_mask)
        
        if compute_spatial_features:
            # Distance to BC edge (not center)
            point_x = coords_t[:, 0]
            
            # Distance to nearest edge
            dist_to_lower = np.abs(point_x - bc_lower)
            dist_to_upper = np.abs(point_x - bc_upper)
            distances_to_edge = np.minimum(dist_to_lower, dist_to_upper)
            
            # Points inside BC have negative distance (signed distance field)
            inside_bc = (point_x >= bc_lower) & (point_x <= bc_upper)
            distances_to_edge[inside_bc] *= -1
            
            series_distances_to_bc_edge.append(distances_to_edge)
            
            # Direction to BC center (still useful for orientation)
            diff = 0 - point_x  
            # Expand to 3D: [x_direction, 0, 0]
            contact_directions = np.zeros_like(coords_t)
            contact_directions[:, 0] = np.sign(diff)  # -1, 0, or 1
            series_contact_directions.append(contact_directions)
        
        
        series_points_t.append(coords_t)
        series_points_tp1.append(coords_tp1)
        series_steps.append(s_tp1)
        series_positions.append(p_tp1)
        series_rotations.append(r_tp1)
        pv_meshes.append(pv_mesh_t)
        pv_meshes_tp1.append(pv_mesh_tp1)
    
    result = {
        'series_id': series_id,
        'points_t': series_points_t,
        'points_tp1': series_points_tp1,
        'steps': series_steps,
        'positions': series_positions,
        'rotations': series_rotations,
        'length': len(series_points_t),
        'meshes': pv_meshes,
        'meshes_tp1' : pv_meshes_tp1
    }
    
    # Add optional features to result
    if compute_bc_mask:
        result['bc_masks'] = series_bc_masks
    if compute_spatial_features:
        result['distances_to_bc_edge'] = series_distances_to_bc_edge
        result['contact_directions'] = series_contact_directions
    
    return result

def extract_data(db_path, total_points, lines):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM strike LIMIT {int(lines)};", conn)

    conn.close()

    all_points_t = []
    all_points_tp1 = []
    all_steps = []
    all_positions = []
    all_rotations = []
    series_lengths = []
    series_ids = []
    press_width = 1.0 #hard coded press_width for press_id = 2 #TODO integrate with DBMS class to get press info on per series basis

    for series_id in tqdm(df['series_id'].unique(), desc="Processing series"):
        group_df = df[df['series_id'] == series_id].reset_index(drop=True)
        series_lengths.append(len(group_df) - 1)
        series_ids.append(series_id)
        # return(group_df)
        # Loop over i and i+1 pairs
        for i in tqdm(range(len(group_df) - 1), desc=f"Series {series_id}", leave=False):
            row_t = group_df.loc[i]
            row_tp1 = group_df.loc[i + 1]

            # Input mesh (coords from frame i)
            mesh_data_t = json.loads(row_t["result"])
            mesh_data_tp1 = json.loads(row_tp1["result"])
            # Get data from frame i+1
            vertices_tp1 = mesh_data_tp1["Vertices"]
            triangles_tp1 = mesh_data_tp1["Triangles"]

            vertices_t = mesh_data_t["Vertices"]
            triangles_t = mesh_data_t["Triangles"]
            tmp_mesh_t = MeshContainer.from_db(vertices_t, triangles_t)
            pv_mesh_t = meshcontainer_to_pv(tmp_mesh_t)
            s_tp1 = np.sum(np.array(mesh_data_tp1["Steps"]))
            p_tp1 = json.loads(row_tp1["position"])

            r_tp1 = json.loads(row_tp1["rotation"])

            try:
                tri_mask, _  = triangle_mask_from_window(pv_mesh_t, center=-p_tp1[0], window_length=press_width, bc_length=3*press_width)
                coords_t, point_triangle_ids, bary_coords = barycentric_sampling(pv_mesh_t, total_points, tri_mask=tri_mask)


                tmp_mesh_tp1 = MeshContainer.from_db(vertices_tp1, triangles_tp1)
                pv_mesh_tp1 = meshcontainer_to_pv(tmp_mesh_tp1)
                coords_tp1 = update_barycentric_points(pv_mesh_tp1, point_triangle_ids, bary_coords)
            except:
                continue
            
            #Normalize point coordiantes

            # coords_t = untransform_points(coords_t, r_tp1, -np.array(p_tp1))
            # coords_tp1 = untransform_points(coords_tp1, r_tp1, -np.array(p_tp1))

            all_points_t.append(coords_t)
            all_points_tp1.append(coords_tp1)
            all_steps.append(s_tp1) # TODO - is this correct we are appending the action of the next step ? 
            all_positions.append(p_tp1) 
            all_rotations.append(r_tp1)

    return [np.array(all_points_t), 
            np.array(all_points_tp1), 
            np.array(all_steps).reshape(-1, 1), 
            np.array(all_positions), 
            np.array(all_rotations), 
            series_lengths, 
            series_ids]

def n_extract_data(db_path, total_points, lines, n_workers=None, 
                   compute_bc_mask=False, compute_spatial_features=False):
    """
    Extract data with parallel processing
    Args:
        db_path: Path to database
        total_points: Number of points to sample
        lines: Number of lines to read from DB
        n_workers: Number of parallel workers (None = use all CPUs)
        compute_bc_mask: If True, compute boundary condition masks
        compute_spatial_features: If True, compute spatial features (distance, direction)
    """
    # Read data from database
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM strike LIMIT {int(lines)};", conn)
    conn.close()
    
    # Prepare arguments for each series
    press_width = 1.0
    series_ids = df['series_id'].unique()
    args_list = [
        (series_id, df[df['series_id'] == series_id].reset_index(drop=True), total_points, press_width)
        for series_id in series_ids
    ]
    
    # Process in parallel - need to pass compute flags
    if n_workers is None:
        n_workers = cpu_count()
    
    print(f"Processing {len(series_ids)} series using {n_workers} workers...")
    
    # Create partial function with compute flags
    from functools import partial
    process_func = partial(process_single_series, 
                          compute_bc_mask=compute_bc_mask,
                          compute_spatial_features=compute_spatial_features)
    
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
        'steps': [],
        'positions': [],
        'rotations': [],
        'series_lengths': [],
        'series_ids': [],
        'meshes' : [],
        'meshes_tp1' : []
    }
    
    # Initialize optional keys if requested
    if compute_bc_mask:
        output['bc_masks'] = []
    if compute_spatial_features:
        output['distances_to_bc_edge'] = []
        output['contact_directions'] = []
    
    # Combine results
    for result in results:
        if result['length'] > 0:  # Only add series that produced data
            output['coords_t'].extend(result['points_t'])
            output['coords_tp1'].extend(result['points_tp1'])
            output['steps'].extend(result['steps'])
            output['positions'].extend(result['positions'])
            output['rotations'].extend(result['rotations'])
            output['series_lengths'].append(result['length'])
            output['series_ids'].append(result['series_id'])
            output['meshes'].extend(result['meshes'])
            output['meshes_tp1'].extend(result['meshes_tp1'])

            
            # Add optional features if present
            if compute_bc_mask:
                output['bc_masks'].extend(result['bc_masks'])
            if compute_spatial_features:
                output['distances_to_bc_edge'].extend(result['distances_to_bc_edge'])
                output['contact_directions'].extend(result['contact_directions'])
    
    # Convert to numpy arrays
    output['coords_t'] = np.array(output['coords_t'])
    output['coords_tp1'] = np.array(output['coords_tp1'])
    output['steps'] = np.array(output['steps']).reshape(-1, 1)
    output['positions'] = np.array(output['positions'])
    output['rotations'] = np.array(output['rotations'])
    
    if compute_bc_mask:
        output['bc_masks'] = np.array(output['bc_masks'])
    if compute_spatial_features:
        output['distances_to_bc_edge'] = np.array(output['distances_to_bc_edge'])
        output['contact_directions'] = np.array(output['contact_directions'])
    
    return output
