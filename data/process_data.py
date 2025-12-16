import sqlite3
import pandas as pd
import numpy as np
import json
import pyvista as pv
import math
from tqdm import tqdm
from scipy.spatial.transform import Rotation


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

            coords_t = untransform_points(coords_t, r_tp1, -np.array(p_tp1))
            coords_tp1 = untransform_points(coords_tp1, r_tp1, -np.array(p_tp1))

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

from multiprocessing import Pool, cpu_count
import sqlite3
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

def process_single_series(args):
    """Process a single series - this will run in parallel"""
    series_id, group_df, total_points, press_width = args
    
    series_points_t = []
    series_points_tp1 = []
    series_steps = []
    series_positions = []
    series_rotations = []
    
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
        
        s_tp1 = np.sum(np.array(mesh_data_tp1["Steps"]))
        p_tp1 = json.loads(row_tp1["position"])
        r_tp1 = json.loads(row_tp1["rotation"])
        
        try:
            tri_mask, _ = triangle_mask_from_window(
                pv_mesh_t, center = -p_tp1[0], 
                window_length=press_width, 
                bc_length=3*press_width
            )
            coords_t, point_triangle_ids, bary_coords = barycentric_sampling(
                pv_mesh_t, total_points, tri_mask=tri_mask
            )
            
            tmp_mesh_tp1 = MeshContainer.from_db(vertices_tp1, triangles_tp1)
            pv_mesh_tp1 = meshcontainer_to_pv(tmp_mesh_tp1)

            coords_tp1 = update_barycentric_points(pv_mesh_tp1, point_triangle_ids, bary_coords)
        except:
            continue
        
        # Normalize point coordinates
        coords_t = untransform_points(coords_t, r_tp1, -np.array(p_tp1))
        coords_tp1 = untransform_points(coords_tp1, r_tp1, -np.array(p_tp1))

        
        series_points_t.append(coords_t)
        series_points_tp1.append(coords_tp1)
        series_steps.append(s_tp1)
        series_positions.append(p_tp1)
        series_rotations.append(r_tp1)
    
    return {
        'series_id': series_id,
        'points_t': series_points_t,
        'points_tp1': series_points_tp1,
        'steps': series_steps,
        'positions': series_positions,
        'rotations': series_rotations,
        'length': len(series_points_t)
    }


def n_extract_data(db_path, total_points, lines, n_workers=None):
    """
    Extract data with parallel processing
    
    Args:
        db_path: Path to database
        total_points: Number of points to sample
        lines: Number of lines to read from DB
        n_workers: Number of parallel workers (None = use all CPUs)
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
    
    # Process in parallel
    if n_workers is None:
        n_workers = cpu_count()
    
    print(f"Processing {len(series_ids)} series using {n_workers} workers...")
    
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_series, args_list),
            total=len(args_list),
            desc="Processing series"
        ))
    
    # Combine results
    all_points_t = []
    all_points_tp1 = []
    all_steps = []
    all_positions = []
    all_rotations = []
    series_lengths = []
    series_ids_out = []
    
    for result in results:
        if result['length'] > 0:  # Only add series that produced data
            all_points_t.extend(result['points_t'])
            all_points_tp1.extend(result['points_tp1'])
            all_steps.extend(result['steps'])
            all_positions.extend(result['positions'])
            all_rotations.extend(result['rotations'])
            series_lengths.append(result['length'])
            series_ids_out.append(result['series_id'])
    
    return [
        np.array(all_points_t),
        np.array(all_points_tp1),
        np.array(all_steps).reshape(-1, 1),
        np.array(all_positions),
        np.array(all_rotations),
        series_lengths,
        series_ids_out
    ]

