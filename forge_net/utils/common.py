import numpy as np
import pyvista as pv
from pathlib import Path

from forge_net.utils.math import quat_to_eulerxyz

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

def actions_from_feature_map(action_features, data):
    # Define all possible features
    feature_map = {
        "steps": lambda: data['steps'],
        "positions": lambda: data['positions'][:, 0].reshape(-1, 1), #if positions we only care about translation in X
        "rotations": lambda: np.array([[quat_to_eulerxyz(quat)[0]] for quat in data['rotations']]) #if rotations we only care about rotation about x
    }
    return np.hstack([feature_map[f]() for f in action_features])


def pyvista_to_open3d(pv_mesh):
    import open3d as o3d
    vertices = pv_mesh.points
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
        
    return o3d_mesh



