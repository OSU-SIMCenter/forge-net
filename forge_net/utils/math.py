import pyvista as pv
import math
import numpy as np
from scipy.sparse import coo_matrix, vstack, eye
from scipy.sparse.linalg import lsqr
import math
from scipy.spatial.transform import Rotation

def barycentric_sampling(mesh: pv.PolyData, num_points: int, tri_mask: np.array = None, seed: int = None) -> tuple[np.array, np.array, np.array]:
    '''
    Returns sampled points and their barycentric information.
    
    Args:
        mesh: Input mesh
        num_points: Number of points to sample
        tri_mask: Optional mask for triangles to sample from
        seed: Random seed for reproducible sampling
    
    Returns:
        points: (N, 3) sampled point positions
        triangle_ids: (N,) which triangle each point belongs to (in original mesh indexing)
        barycentric_coords: (N, 3) barycentric coordinates (b0, b1, b2)
    '''
    # Set random seed if provided
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
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
            point_translations.append([rng.random(), rng.random()])
            point_triangle_ids.append(i)
    
    for i in range(num_points - len(point_translations)):
        point_translations.append([rng.random(), rng.random()])
        point_triangle_ids.append(rng.randint(0, num_triangles))
    
    # Compute points and barycentric coordinates
    sampled_points = []
    barycentric_coords = []
    global_triangle_ids = []
    
    for i in range(len(point_triangle_ids)):
        tri_id = point_triangle_ids[i]
        idx0, idx1, idx2 = triangles[tri_id]
        
        v0 = points[idx0]  # A
        v1 = points[idx1]  # B
        v2 = points[idx2]  # C
        
        r0, r1 = point_translations[i]
        
        b0 = 1 - math.sqrt(r0)
        b1 = math.sqrt(r0) * (1 - r1)
        b2 = r1 * math.sqrt(r0)
        
        point = (b0 * v0) + (b1 * v1) + (b2 * v2)
        
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

def get_graph_laplacian(mesh):
    num_vertices = mesh.n_points
    faces = mesh.regular_faces  # Shape (N, 3)
    
    # Extract all edges from the triangles
    # Each face (v1, v2, v3) gives 3 edges: (v1,v2), (v2,v3), (v3,v1)
    v1 = faces[:, 0]
    v2 = faces[:, 1]
    v3 = faces[:, 2]
    
    # We want undirected edges, so we add both directions
    rows = np.concatenate([v1, v2, v2, v3, v3, v1])
    cols = np.concatenate([v2, v1, v3, v2, v1, v3])
    
    # Weight of 1 for every connected edge
    data = np.ones_like(rows, dtype=float)
    
    # Create the Adjacency/Weight matrix W
    # sum_duplicates=True (default) handles vertices shared by multiple triangles
    W = coo_matrix((data, (rows, cols)), shape=(num_vertices, num_vertices)).tocsr()
    
    # Ensure the weights are binary (1 if connected, 0 if not)
    # even if an edge is shared by multiple faces
    W.data[:] = 1.0
    
    # D is the Degree matrix (diagonal)
    # The degree is the number of neighbors for each vertex
    degree = np.array(W.sum(axis=1)).flatten()
    D = coo_matrix((degree, (np.arange(num_vertices), np.arange(num_vertices))), 
                   shape=(num_vertices, num_vertices)).tocsr()
    
    # L = D - W
    return D - W

def invert_deltas_with_smoothness(predicted_deltas, triangle_ids, barycentric_coords, mesh, alpha=0.1, damp=1e-4):
    num_samples = len(predicted_deltas)
    num_vertices = mesh.n_points
    triangles = mesh.regular_faces
    
    # 1. Build the Barycentric Matrix A
    rows, cols, weights = [], [], []
    for i in range(num_samples):
        v_indices = triangles[triangle_ids[i]]
        w = barycentric_coords[i]
        for j in range(3):
            rows.append(i)
            cols.append(v_indices[j])
            weights.append(w[j])
            
    A = coo_matrix((weights, (rows, cols)), shape=(num_samples, num_vertices)).tocsr()
    
    # 2. Build Laplacian L and Augment
    if alpha > 0:
        L = get_graph_laplacian(mesh)
        # Combine: A_augmented = [A; sqrt(alpha) * L]
        A_augmented = vstack([A, np.sqrt(alpha) * L])
        # P_augmented = [P; 0]
        padding = np.zeros((num_vertices, 3))
        rhs_all = np.vstack([predicted_deltas, padding])
    else:
        A_augmented = A
        rhs_all = predicted_deltas

    # 3. Solve per axis
    vertex_deltas = np.zeros((num_vertices, 3))
    for d in range(3):
        # rhs_all[:, d] handles X, Y, and Z independently
        res = lsqr(A_augmented, rhs_all[:, d], damp=damp)
        vertex_deltas[:, d] = res[0]
        
    return vertex_deltas

def normalize_points(points):
    point_min = np.min(points, axis=0)
    point_max = np.max(points, axis=0)
    return( (points - point_min) / (point_max - point_min) )

def compute_haussdorff_distance(pv_mesh1, pv_mesh2, samples=1000):
    from forge_net.utils.common import pyvista_to_open3d
    mesh1 = pyvista_to_open3d(pv_mesh1)
    mesh2 = pyvista_to_open3d(pv_mesh2)
    pcd1 = mesh1.sample_points_uniformly(number_of_points=samples)
    pcd2 = mesh2.sample_points_uniformly(number_of_points=samples)
    np.asarray(pcd1.points)
    np.asarray(pcd2.points)

    dists_1to2 = pcd1.compute_point_cloud_distance(pcd2)
    dists_2to1 = pcd2.compute_point_cloud_distance(pcd1)

    # chamfer_dist = np.mean(np.square(dists_1to2)) + np.mean(np.square(dists_2to1))
    hausdorff_dist = max(max(dists_1to2), max(dists_2to1))  # https://en.wikipedia.org/wiki/Hausdorff_distance
    return(hausdorff_dist)
 

def transform_points(points, quaternion, translation_vector):
    return Rotation.from_quat(quaternion).apply(points) + translation_vector

def untransform_points(points, quaternion, translation_vector):
    return Rotation.from_quat(quaternion).inv().apply(np.array(points) - np.array(translation_vector))

def quat_to_eulerxyz(quaternion):
    return Rotation.from_quat(quaternion).as_euler('xyz',degrees=True)

def eulerxyz_to_quat(xyz_degtuple):
    return Rotation.from_euler('xyz',xyz_degtuple,degrees=True).as_quat()