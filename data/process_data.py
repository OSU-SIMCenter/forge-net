import sqlite3
import pandas as pd
import numpy as np
import json
import pyvista as pv
import math
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from multiprocessing import Pool, cpu_count

class MeshContainer:
    def __init__(self, vertices, triangles):
        self.vertices = np.array(vertices).reshape(-1, 3)
        self.triangles = np.array(triangles).reshape(-1, 3)

    @classmethod
    def from_db(cls, vertices, triangles):
        return cls(vertices, triangles)

def meshcontainer_to_pv(mesh):
    """
    Convert a MeshContainer instance to a PyVista PolyData mesh.
    mesh: MeshContainer with .vertices (N, 3) and .triangles (M, 3)
    """
    n_faces = len(mesh.triangles)
    face_array = np.hstack([
        np.full((n_faces, 1), 3),
        mesh.triangles
    ]).astype(np.int64)

    return pv.PolyData(mesh.vertices, face_array)

def barycentric_sampling(mesh: pv.PolyData, num_points: int, tri_mask: np.array = None) -> tuple[np.array, np.array, np.array]:
    '''
    Returns sampled points and their barycentric information.
    
    Returns:
        points: (N, 3) sampled point positions
        triangle_ids: (N,) which triangle each point belongs to (in original mesh indexing)
        barycentric_coords: (N, 3) barycentric coordinates (b0, b1, b2)
    '''
    mesh = mesh.compute_cell_sizes()
    triangles = mesh.regular_faces
    triangle_areas = mesh.cell_data["Area"]
    points = mesh.points
    
    # Store original triangle indices before masking
    original_tri_indices = np.arange(len(triangles))
    
    if tri_mask is not None:
        triangles = triangles[tri_mask]
        triangle_areas = triangle_areas[tri_mask]
        original_tri_indices = original_tri_indices[tri_mask]
        total_area = np.sum(triangle_areas)
    else:
        total_area = mesh.area
    
    num_triangles = len(triangle_areas)
    assert num_triangles > 0, "Triangle mask is empty"
    
    point_translations = []
    point_triangle_ids = []  # Indices into masked triangles
    
    for i in range(num_triangles):
        for _ in range(math.floor(triangle_areas[i] / total_area * num_points)):
            point_translations.append([np.random.random(), np.random.random()])
            point_triangle_ids.append(i)
    
    for i in range(num_points - len(point_translations)):
        point_translations.append([np.random.random(), np.random.random()])
        point_triangle_ids.append(np.random.randint(0, num_triangles))
    
    # Compute points and barycentric coordinates
    sampled_points = []
    barycentric_coords = []
    global_triangle_ids = []
    
    for i in range(len(point_triangle_ids)):
        tri_id = point_triangle_ids[i]
        idx0, idx1, idx2 = triangles[tri_id]
        
        v0 = points[idx0]
        v1 = points[idx1]
        v2 = points[idx2]
        
        r0, r1 = point_translations[i]
        b0 = 1 - math.sqrt(r0)
        b1 = math.sqrt(r0) * (1 - r1)
        b2 = r1 * math.sqrt(r0)
        
        point = b0 * v0 + b1 * v1 + b2 * v2
        sampled_points.append(point)
        barycentric_coords.append([b0, b1, b2])
        global_triangle_ids.append(original_tri_indices[tri_id])
    
    return np.array(sampled_points), np.array(global_triangle_ids), np.array(barycentric_coords)

def update_barycentric_points(deformed_mesh: pv.PolyData, triangle_ids: np.array, barycentric_coords: np.array) -> np.array:
    '''
    Updates point positions based on deformed mesh using stored barycentric coordinates.
    
    Args:
        deformed_mesh: Deformed mesh with same topology as original
        triangle_ids: (N,) triangle indices for each point
        barycentric_coords: (N, 3) barycentric coordinates
    
    Returns:
        (N, 3) updated point positions
    '''
    triangles = deformed_mesh.regular_faces
    vertices = deformed_mesh.points
    
    updated_points = []
    for i in range(len(triangle_ids)):
        tri_id = triangle_ids[i]
        idx0, idx1, idx2 = triangles[tri_id]
        
        v0 = vertices[idx0]
        v1 = vertices[idx1]
        v2 = vertices[idx2]
        
        b0, b1, b2 = barycentric_coords[i]
        point = b0 * v0 + b1 * v1 + b2 * v2
        updated_points.append(point)
    
    return np.array(updated_points)

def triangle_mask_from_window(mesh: pv.PolyData, center: float, window_length: float, bc_length: float = 0.0) -> np.ndarray:
    '''
    Creates a boolean mask for triangles based on their centroid's x-coordinate.
    
    Args:
        mesh: PyVista PolyData mesh
        center: Center of the window
        window_length: Length of the window
        bc_length: Additional boundary condition length
    
    Returns:
        Boolean array of shape (n_triangles,) indicating which triangles fall in the window
    '''
    # Get triangle centroids
    triangles = mesh.regular_faces
    vertices = mesh.points
    
    # Calculate centroid x-coordinates for each triangle
    triangle_centroids_x = np.mean(vertices[triangles, 0], axis=1)
    
    lower_bound = center - (window_length / 2 + bc_length / 2)
    upper_bound = center + (window_length / 2 + bc_length / 2)
    
    bounds = [lower_bound, upper_bound]
    
    mask = (triangle_centroids_x >= lower_bound) & (triangle_centroids_x <= upper_bound)
    
    return mask, bounds

def normalize_points(points):
    point_min = np.min(points, axis=0)
    point_max = np.max(points, axis=0)
    return( (points - point_min) / (point_max - point_min) )

def transform_points(points, quaternion, translation_vector):
    return Rotation.from_quat(quaternion).apply(points) + translation_vector

def untransform_points(points, quaternion, translation_vector):
    return Rotation.from_quat(quaternion).inv().apply(np.array(points) - np.array(translation_vector))

def quat_to_eulerxyz(quaternion):
    return Rotation.from_quat(quaternion).as_euler('xyz',degrees=True)

def eulerxyz_to_quat(xyz_degtuple):
    return Rotation.from_euler('xyz',xyz_degtuple,degrees=True).as_quat()

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
        mesh_data_tp1 = json.loads(row_tp1["result"])
        vertices_tp1 = mesh_data_tp1["Vertices"]
        triangles_tp1 = mesh_data_tp1["Triangles"]
        vertices_t = mesh_data_t["Vertices"]
        triangles_t = mesh_data_t["Triangles"]
        
        tmp_mesh_t = MeshContainer.from_db(vertices_t, triangles_t)
        pv_mesh_t = meshcontainer_to_pv(tmp_mesh_t)
        tmp_mesh_tp1 = MeshContainer.from_db(vertices_tp1, triangles_tp1)
        pv_mesh_tp1 = meshcontainer_to_pv(tmp_mesh_tp1)
        
        s_tp1 = np.sum(np.array(mesh_data_tp1["Steps"]))
        p_tp1 = json.loads(row_tp1["position"])
        r_tp1 = json.loads(row_tp1["rotation"])
        
        # Normalize point coordinates
        # pv_mesh_t.points = untransform_points(np.array(pv_mesh_t.points), [0,0,0,0], -np.array(p_tp1))
        # pv_mesh_tp1.points = untransform_points(np.array(pv_mesh_tp1.points), [0,0,0,0], -np.array(p_tp1))
        # pv_mesh_t.points = transform_points(np.array(pv_mesh_t.points), r_tp1, np.array(p_tp1))
        # pv_mesh_tp1.points = transform_points(np.array(pv_mesh_tp1.points), r_tp1, np.array(p_tp1))
        # pv_mesh_t.points += np.array(p_tp1)
        # pv_mesh_tp1.points += np.array(p_tp1)        

        try:
            tri_mask, _ = triangle_mask_from_window(
                pv_mesh_t, center = 0.0, 
                window_length=press_width, 
                bc_length=3*press_width
            )
            coords_t, point_triangle_ids, bary_coords = barycentric_sampling(
                pv_mesh_t, total_points, tri_mask=tri_mask
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

