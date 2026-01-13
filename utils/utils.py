import numpy as np
import pyvista as pv
from pathlib import Path
import math
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

def get_project_root() -> Path:
    return Path(__file__).parent.parent

ROOT = get_project_root()

def get_datasets_path() -> Path:
    return(ROOT / 'data' / 'datasets')

def get_models_path() -> Path:
    return(ROOT / 'model' / 'saved_models')

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

