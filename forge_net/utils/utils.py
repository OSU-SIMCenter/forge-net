import numpy as np
import pyvista as pv
import trimesh as tm
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

import numpy as np

def barycentric_sampling(
    vertices: np.ndarray,
    triangles: np.ndarray,
    num_points: int,
):
    """
    vertices: (V,3)
    triangles: (T,3)
    returns:
        sampled_points: (N,3)
        triangle_ids: (N,)
        barycentric_coords: (N,3)
    """

    # --- gather triangle vertices ---
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    # --- compute triangle areas ---
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    probs = areas / areas.sum()

    # --- randomly choose triangles weighted by area ---
    tri_ids = np.random.choice(len(triangles), size=num_points, p=probs)

    # --- sample barycentric coords ---
    r = np.random.rand(num_points, 2)
    sqrt_r0 = np.sqrt(r[:, 0])

    b0 = 1 - sqrt_r0
    b1 = sqrt_r0 * (1 - r[:, 1])
    b2 = sqrt_r0 * r[:, 1]

    # --- gather chosen triangle vertices ---
    v0_sel = v0[tri_ids]
    v1_sel = v1[tri_ids]
    v2_sel = v2[tri_ids]

    sampled_points = (
        b0[:, None] * v0_sel +
        b1[:, None] * v1_sel +
        b2[:, None] * v2_sel
    )

    bary = np.stack([b0, b1, b2], axis=1)

    return sampled_points.astype(np.float32), tri_ids, bary.astype(np.float32)

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

def tm_barycentric_sampling(pv_mesh: pv.PolyData, 
                            num_points: int, 
                            tri_mask: np.array = None,
                            seed: int = None) -> tuple[np.array, np.array, np.array]:
    
    if tri_mask is not None:
        faces = pv_mesh.regular_faces[tri_mask]
        original_tri_indices = np.arange(len(pv_mesh.regular_faces))[tri_mask]
    else:
        faces = pv_mesh.regular_faces
        original_tri_indices = np.arange(len(pv_mesh.regular_faces))

    tm_mesh = tm.Trimesh(vertices=pv_mesh.points, faces=faces)
    points, tri_ids_local = tm.sample.sample_surface(tm_mesh, count=num_points, seed=seed)
    tri_ids_global = original_tri_indices[tri_ids_local]

    bary_coords = tm.triangles.points_to_barycentric(
                        triangles=tm_mesh.triangles[tri_ids_local],
                        points=points)


    return(points, tri_ids_global, bary_coords)

def tm_update_barycentric_points(deformed_mesh: pv.PolyData, triangle_ids: np.array, bary_coords: np.array):
    deformed_tm = tm.Trimesh(
        vertices=deformed_mesh.points,
        faces=deformed_mesh.regular_faces
    )

    updated_points = tm.triangles.barycentric_to_points(
        triangles=deformed_tm.triangles[triangle_ids],
        barycentric=bary_coords
    )
    return np.array(updated_points)

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

