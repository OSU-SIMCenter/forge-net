import numpy as np
import pyvista as pv
from pathlib import Path

from forge_net.utils.math import quat_to_eulerxyz, transform_points, untransform_points

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


from dataclasses import dataclass, field
@dataclass
class PressInfo:
    name: str = ""
    width: int = 1
    height: int = 3
    direction: list[float] = field(default_factory=lambda: [0, 1, 0])

    def get_dict(self) -> "dict[str,object]":
        return {"Name": self.name, "Width": self.width, "Height": self.height}


def get_tool_mesh(points, translation, rotation, press_info=None, num_presses=2, ref='tool'):
    
    if press_info is None:
        press_info = PressInfo()
    
    planes = []
    directions = np.array([0, 1, 2])
    press_dir = np.array(press_info.direction)
    # apply transformation to mesh
    transformed_points = transform_points(points, rotation, translation)
    # find the directions we want to check against 
    # (if direction is [0,1,0] we want to check indices 0,2 ignoring y)
    directions_to_check = directions[~np.array(press_dir, dtype=bool)]
    points_in_range = transformed_points[
        (transformed_points[:, directions_to_check[0]] >= -1 * press_info.width / 2)
        & (transformed_points[:, directions_to_check[0]] <= press_info.width / 2)
        & (transformed_points[:, directions_to_check[1]] >= -1 * press_info.height / 2)
        & (transformed_points[:, directions_to_check[1]] <= press_info.height / 2)
    ]
    direction_idx = press_info.direction.index(1)
    vals = points_in_range[:, direction_idx]
    max_points = max(vals)
    # get plane mesh
    offset = press_dir * 0.01
    planes.append(
        pv.Plane(
            center=press_dir * max_points + offset,
            direction=press_info.direction,
            i_size=press_info.height,
            j_size=press_info.width,
        )
    )
    if num_presses == 2:
        min_points = min(vals)
        planes.append(
            pv.Plane(
                center=press_dir * min_points - offset,
                direction=press_info.direction,
                i_size=press_info.height,
                j_size=press_info.width,
            )
        )
    
    if ref == 'tool': # rotate the tools
        for plane in planes:
            plane.points = untransform_points(plane.points, rotation, translation)
    
    return planes

def pyvista_to_open3d(pv_mesh):
    import open3d as o3d
    vertices = pv_mesh.points
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
        
    return o3d_mesh



