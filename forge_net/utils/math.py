import pyvista as pv
import math
import numpy as np
from scipy.sparse import coo_matrix, vstack, eye
from scipy.sparse.linalg import lsqr
import math
from scipy.spatial.transform import Rotation
from scipy.interpolate import RBFInterpolator

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

def tetrahedral_barycentric_sampling(mesh: pv.UnstructuredGrid, 
                                        num_points: int, tet_mask: np.ndarray = None, 
                                        node_features: np.ndarray = None, seed: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Returns sampled points inside a tetrahedral mesh and their barycentric information.
    
    Args:
        mesh: Input pyvista UnstructuredGrid (pure tetrahedra)
        num_points: Number of points to sample
        tet_mask: Optional mask for tetrahedra to sample from
        seed: Random seed for reproducible sampling
    
    Returns:
        points: (N, 3) sampled point positions
        global_tet_ids: (N,) which tetrahedron each point belongs to (in original mesh indexing)
        barycentric_coords: (N, 4) barycentric coordinates (b0, b1, b2, b3)
    '''
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # Compute volume instead of area
    mesh = mesh.compute_cell_sizes()
    
    # Extract tetrahedral connectivity. Assuming pure tet mesh: [4, v0, v1, v2, v3, 4, ...]
    cells = mesh.cells.reshape(-1, 5)
    tetrahedra = cells[:, 1:] 
    
    tet_volumes = np.abs(mesh.cell_data["Volume"])
    points = mesh.points
    
    # Store original tet indices before masking
    original_tet_indices = np.arange(len(tetrahedra))
    
    if tet_mask is not None:
        tetrahedra = tetrahedra[tet_mask]
        tet_volumes = tet_volumes[tet_mask]
        original_tet_indices = original_tet_indices[tet_mask]
        
    total_volume = np.sum(tet_volumes)
    num_tetrahedra = len(tet_volumes)
    assert num_tetrahedra > 0, "Tetrahedral mask is empty"
    
    # --- 1. Distribute points based on volume ---
    # Vectorized equivalent of the nested floor loop
    counts = np.floor((tet_volumes / total_volume) * num_points).astype(int)
    point_tet_ids = np.repeat(np.arange(num_tetrahedra), counts).tolist()
    
    # Assign remaining points randomly
    remainder = num_points - len(point_tet_ids)
    if remainder > 0:
        point_tet_ids.extend(rng.randint(0, num_tetrahedra, size=remainder))
        
    point_tet_ids = np.array(point_tet_ids)
    
    # --- 2. Compute 3D Barycentric Coordinates ---
    # We need 3 random variables for a 3D simplex (tetrahedron)
    r0 = rng.random(num_points)
    r1 = rng.random(num_points)
    r2 = rng.random(num_points)
    
    # Mathematical roots for uniform 3D distribution
    r0_cbrt = np.cbrt(r0)
    r1_sqrt = np.sqrt(r1)
    
    b0 = 1.0 - r0_cbrt
    b1 = r0_cbrt * (1.0 - r1_sqrt)
    b2 = r0_cbrt * r1_sqrt * (1.0 - r2)
    b3 = r0_cbrt * r1_sqrt * r2
    
    barycentric_coords = np.column_stack((b0, b1, b2, b3))
    
    # --- 3. Compute final point positions ---
    # Fetch the 4 vertices for every selected tetrahedron
    tet_indices = tetrahedra[point_tet_ids]
    
    v0 = points[tet_indices[:, 0]]
    v1 = points[tet_indices[:, 1]]
    v2 = points[tet_indices[:, 2]]
    v3 = points[tet_indices[:, 3]]
    
    # Apply weights using broadcasting
    sampled_points = (b0[:, None] * v0 + 
                      b1[:, None] * v1 + 
                      b2[:, None] * v2 + 
                      b3[:, None] * v3)
    
    global_tet_ids = original_tet_indices[point_tet_ids]
    
    # --- NEW: Compute interpolated features ---
    sampled_features = None
    if node_features is not None:
        f0 = node_features[tet_indices[:, 0]]
        f1 = node_features[tet_indices[:, 1]]
        f2 = node_features[tet_indices[:, 2]]
        f3 = node_features[tet_indices[:, 3]]
        
        # Handle both 1D (e.g., single temperature value) and 2D features (e.g., RGB colors, vectors)
        if node_features.ndim == 1:
            sampled_features = b0 * f0 + b1 *f1 + b2 * f2 + b3 * f3
        else:
            sampled_features = b0[:, None] * f0 + b1[:, None] * f1 + b2[:, None] * f2 + b3[:, None] * f3

    return sampled_points, global_tet_ids, barycentric_coords, sampled_features

def update_tetrahedral_barycentric_points(
    deformed_mesh: pv.UnstructuredGrid, 
    tet_ids: np.ndarray, 
    barycentric_coords: np.ndarray,
    node_features: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Updates point positions and features based on a deformed tetrahedral mesh 
    using stored barycentric coordinates.
    
    Args:
        deformed_mesh: Deformed mesh (pyvista.UnstructuredGrid).
        tet_ids: (N,) array of tetrahedron indices for each point.
        barycentric_coords: (N, 4) array of barycentric coordinates.
        node_features: (V,) or (V, M) optional array of vertex features for the deformed mesh.
    
    Returns:
        updated_points: (N, 3) updated spatial positions.
        updated_features: (N,) or (N, M) updated features (or None if node_features not provided).
    '''
    # Extract the connectivity array for pure tetrahedra
    cells = deformed_mesh.cells.reshape(-1, 5)
    tetrahedra = cells[:, 1:] 
    
    # Extract all vertex positions
    vertices = deformed_mesh.points
    
    # Get the vertex indices for the specific tetrahedra containing our points
    tet_indices = tetrahedra[tet_ids]
    
    # Fetch the 3D coordinates of the 4 vertices for each tetrahedron
    v0 = vertices[tet_indices[:, 0]]
    v1 = vertices[tet_indices[:, 1]]
    v2 = vertices[tet_indices[:, 2]]
    v3 = vertices[tet_indices[:, 3]]
    
    # Extract the barycentric weights
    # We create a 1D version for scalar features, and a 2D version for spatial broadcasting
    b0_flat = barycentric_coords[:, 0]
    b1_flat = barycentric_coords[:, 1]
    b2_flat = barycentric_coords[:, 2]
    b3_flat = barycentric_coords[:, 3]
    
    b0_vec = b0_flat[:, None]
    b1_vec = b1_flat[:, None]
    b2_vec = b2_flat[:, None]
    b3_vec = b3_flat[:, None]
    
    # Compute the new positions simultaneously using vectorized addition
    updated_points = b0_vec * v0 + b1_vec * v1 + b2_vec * v2 + b3_vec * v3
    
    # --- Compute updated features ---
    updated_features = None
    if node_features is not None:
        f0 = node_features[tet_indices[:, 0]]
        f1 = node_features[tet_indices[:, 1]]
        f2 = node_features[tet_indices[:, 2]]
        f3 = node_features[tet_indices[:, 3]]
        
        if node_features.ndim == 1:
            # For 1D scalar features (e.g., temperature)
            updated_features = (b0_flat * f0) + (b1_flat * f1) + (b2_flat * f2) + (b3_flat * f3)
        else:
            # For multi-dimensional features (e.g., RGB colors, velocity vectors)
            updated_features = (b0_vec * f0) + (b1_vec * f1) + (b2_vec * f2) + (b3_vec * f3)
            
    return updated_points, updated_features

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

def rbf_interpolate_deformation(sparse_old, sparse_new, dense_old):
    '''
    sparse_old: e.g (1024, 3) - Original positions of known points
    sparse_new: e.g (1024, 3) - Deformed positions of known points
    dense_old:  e.g (10000, 3) - Original positions of all sampled points
    '''
    # 1. Calculate the displacement (the "delta")
    displacements = sparse_new - sparse_old
    print(displacements.shape)
    print(sparse_new.shape)
    print(sparse_old.shape)
    print(dense_old.shape)
    
    # 2. Fit the RBF to learn the mapping: Position -> Displacement
    # 'thin_plate_spline' is excellent for smooth surface deformations
    interpolator = RBFInterpolator(sparse_old, displacements, kernel='thin_plate_spline')
    
    # 3. Predict the displacement for the 10,000 points
    predicted_deltas = interpolator(dense_old)
    
    # 4. Apply the displacement
    dense_new = dense_old + predicted_deltas
    
    return dense_new

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

def point_cloud_stats(points):
    """
    Prints statistics about a point cloud.
    
    Args:
        points: array-like of shape (N, 3) with columns [x, y, z]
    """
    pts = np.asarray(points)
    assert pts.ndim == 2 and pts.shape[1] == 3, "Input must be (N, 3)"

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    n = len(pts)

    # Bounding box
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ranges = maxs - mins

    # Volume and density
    volume = ranges[0] * ranges[1] * ranges[2]
    density = n / volume if volume > 0 else float("inf")

    # Average nearest-neighbour distance (sampled for large clouds)
    sample_size = min(n, 1000)
    sample = pts[np.random.choice(n, sample_size, replace=False)]
    # Vectorised pairwise distances within the sample
    diff = sample[:, None, :] - sample[None, :, :]          # (S, S, 3)
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))          # (S, S)
    np.fill_diagonal(dist_matrix, np.inf)
    avg_nn_dist = dist_matrix.min(axis=1).mean()

    print("=" * 45)
    print(f"  Point Cloud Statistics  (N = {n:,})")
    print("=" * 45)
    print(f"  {'Axis':<6} {'Min':>10} {'Max':>10} {'Range':>10}")
    print(f"  {'-'*36}")
    for axis, mn, mx, rng in zip("XYZ", mins, maxs, ranges):
        print(f"  {axis:<6} {mn:>10.4f} {mx:>10.4f} {rng:>10.4f}")
    print()
    print(f"  Centroid       x={x.mean():.4f},  y={y.mean():.4f},  z={z.mean():.4f}")
    print(f"  Std dev        x={x.std():.4f},  y={y.std():.4f},  z={z.std():.4f}")
    print()
    print(f"  Bounding volume  {volume:.4f} units³")
    print(f"  Point density    {density:.4f} pts / unit³")
    print(f"  Avg NN distance  {avg_nn_dist:.4f} units  (sample={sample_size:,})")
    print("=" * 45)