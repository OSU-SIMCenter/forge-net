#!/usr/bin/env python3
"""
Unity sends: JSON text
  {
    "request": "update" | "strike" | "heat" | "reset" | "undo",
    "translation": float,
    "rotation": float,
    "force": float
  }

Server replies: binary
  [headerLen (4 bytes, little-endian uint32)]
  [header JSON (UTF-8, headerLen bytes)]
  [vertices (float32 LE)]
  [faces (int32 LE)]
"""

import asyncio
import json
import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

import torch
from forge_net.utils.utils import *
from model.trainer import Trainer
from utils.utils import *
import yaml

from eval import forward
import pysplashsurf as splashsurf


# ---------------------------
# Data model
# ---------------------------

@dataclass
class ClientRequest:
    request: str
    vertices: np.ndarray
    triangles: np.ndarray
    translation: float
    rotation: float
    force: float

    @staticmethod
    def from_json(s: str) -> "ClientRequest":
        obj = json.loads(s)

        return ClientRequest(
            request=str(obj.get("request", "")),
            vertices=np.asarray(obj.get("vertices", []), dtype=np.float32),
            triangles=np.asarray(obj.get("triangles", []), dtype=np.int32),
            translation=float(obj.get("translation", 0.0)),
            rotation=float(obj.get("rotation", 0.0)),
            force=float(obj.get("force", 0.0)),
        )


# ---------------------------
# Update logic
# ---------------------------

def handle_update(
    req: ClientRequest,
    trainer: Trainer,
    cache: dict
) -> Tuple[np.ndarray, np.ndarray, bool]:

    vertices_input = req.vertices.reshape(-1, 3).astype(np.float32)

    triangles = req.triangles.reshape(-1, 3).astype(np.int32)

    # --------------------------------------------------
    # FIRST HIT → compute barycentric sampling once
    # --------------------------------------------------
    if cache["bary"] is None:
        vertices_input[:, [1, 2]] = vertices_input[:, [2, 1]]

        vertices_input[:, 0] *= 4.0
        vertices_input[:, 1] *= 4.0
        vertices_input[:, 2] *= 4.0

        states, tri_ids, bary = barycentric_sampling(
            vertices_input,
            triangles,
            num_points=1000
        )

        

        cache["bary"] = states

    # --------------------------------------------------
    # SUBSEQUENT HITS → reuse bary coords
    # --------------------------------------------------

    states = cache["bary"]


    translation = np.mean(states[:, 0], axis=0)

    states[:, 0] -= translation

    print("min_x: ", np.min(states[:, 0], axis=0))

    print("average_x: ", np.mean(states[:, 0], axis=0))

    print("max_x: ", np.max(states[:, 0], axis=0))

    # --------------------------------------------------
    # Neural network forward
    # --------------------------------------------------
    states_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
    states_tensor = states_tensor.permute(0, 2, 1)

    batch_size = states_tensor.shape[0]
    action_dims = trainer.config["network"]["action_dims"]
    steps = torch.ones((batch_size, action_dims), dtype=torch.float32) * 1

    trainer.net.eval()
    tensor = forward(trainer, states_tensor, steps)

    if tensor.dim() == 3:
        tensor = tensor[0]  # (3, N)

    deltas = tensor.detach().cpu().numpy()

    # Ensure (N, 3)
    if deltas.shape[0] == 3:
        deltas = deltas.T

    deltas = deltas.astype(np.float32)

    states[:, 0] += translation

    deformed_points = states + deltas / 100

    # result = splashsurf.reconstruct_surface(
    #     deformed_points.astype(np.float32),
    #     particle_radius=0.05,
    #     smoothing_length=5,
    #     cube_size=0.5
    # )

    # vertices_out = result.mesh.vertices.astype(np.float32)
    # triangles_out = result.mesh.triangles.astype(np.int32)

    # Invert normals for Unity
    # triangles_out[:, [1, 2]] = triangles_out[:, [2, 1]]

    triangles_out = np.ones((1,3))

    # --------------------------------------------------
    # Update cached mesh for next hit (accumulation)
    # --------------------------------------------------

    cache["bary"] = deformed_points.copy()

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(24,8))

    pc1 = states
    pc2 = deformed_points
    
    ax1 = fig.add_subplot(131, projection = '3d')
    ax1.scatter(xs=pc1[:,0],ys=pc1[:,1],zs=pc1[:,2], s=2.2)
    
    ax2 = fig.add_subplot(132, projection = '3d')
    ax2.scatter(xs=pc2[:,0],ys=pc2[:,1],zs=pc2[:,2], s=2.2, color='red')

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    diff_magnitudes = np.linalg.norm(deltas, axis=1)

    ax3 = fig.add_subplot(133)
    # X-axis: pc1[:, 0] (X coordinates), Y-axis: diff_magnitudes
    ax3.scatter(pc1[:, 0], diff_magnitudes, s=5, color='purple', alpha=0.6)
    ax3.set_title("Magnitude of Difference vs X")
    ax3.set_xlabel("X Coordinate")
    ax3.set_ylabel("Difference Magnitude")
    ax3.grid(True, linestyle='--', alpha=0.6)


    plt.show()

    return cache["bary"], triangles_out, False


# ---------------------------
# Binary protocol
# ---------------------------

def make_binary_reply(vertices, faces, is_pressing):

    v = np.ascontiguousarray(vertices.flatten(), dtype=np.float32)
    f = np.ascontiguousarray(faces.flatten(), dtype=np.int32)

    header_obj = {
        "is_pressing": bool(is_pressing),
        "counts": {
            "vertices": int(v.size),
            "faces": int(f.size),
        },
    }

    header_json = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
    prefix = struct.pack("<I", len(header_json))
    body = v.tobytes(order="C") + f.tobytes(order="C")

    return prefix + header_json + body


# ---------------------------
# WebSocket handler
# ---------------------------

async def client_handler(ws: WebSocketServerProtocol, trainer: Trainer):

    print(f"[connect] {ws.remote_address}")

    # Per-connection cache
    cache = {
        "tri_ids": None,
        "bary": None,
        "base_vertices": None,
    }

    try:
        async for message in ws:

            if isinstance(message, bytes):
                continue

            req = ClientRequest.from_json(message)

            vertices, faces, is_pressing = handle_update(
                req,
                trainer,
                cache
            )

            await ws.send(make_binary_reply(vertices, faces, is_pressing))

    except websockets.ConnectionClosed:
        pass
    except Exception as e:
        print(f"[error] {e}")
    finally:
        print(f"[disconnect] {ws.remote_address}")


# ---------------------------
# Main
# ---------------------------

async def main(host="localhost", port=8765):

    base_path = get_project_root()
    run_name = "mse_1024_unmasked"
    config_path = base_path / "runs" / run_name / "config_out.yml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    trainer = Trainer(config, log_to_tb=False)

    print("Trainer initialized.")
    print(f"Starting server on ws://{host}:{port}")

    async with websockets.serve(
        lambda ws: client_handler(ws, trainer),
        host,
        port,
        max_size=None
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())