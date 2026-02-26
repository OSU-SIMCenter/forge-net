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
  [vertices (float32 LE, count = counts["vertices"])]
  [faces (int32 LE, count = counts["faces"])]
"""

import asyncio
import json
import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

from pathlib import Path
import torch
from forge_net.utils.utils import *
from model.trainer import Trainer
from utils.utils import *
import yaml

from eval import forward
import pysplashsurf as splashsurf


# ---------------------------
# Data model (incoming)
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

        vertices = np.asarray(obj.get("vertices", []), dtype=np.float32)
        triangles = np.asarray(obj.get("triangles", []), dtype=np.int32)

        return ClientRequest(
            request=str(obj.get("request", "")),
            vertices=vertices,
            triangles=triangles,
            translation=float(obj.get("translation", 0.0)),
            rotation=float(obj.get("rotation", 0.0)),
            force=float(obj.get("force", 0.0)),
        )


# ---------------------------
# Application logic
# ---------------------------

def handle_update(req: ClientRequest, trainer: Trainer) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Return (vertices, faces, is_pressing)
    """

    vertices_reshaped = req.vertices.reshape(-1, 3).astype(np.float32)
    triangles = req.triangles.reshape(-1, 3).astype(np.int32)

    # Barycentric sampling
    states, tri_ids, bary = barycentric_sampling(
        vertices_reshaped,
        triangles,
        num_points=1000
    )

    # Convert to tensor
    states = torch.tensor(states, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
    states = states.permute(0, 2, 1)  # (1, 3, N)

    # Prepare dummy action
    batch_size = states.shape[0]
    action_dims = trainer.config["network"]["action_dims"]
    steps = torch.ones((batch_size, action_dims), dtype=torch.float32)

    # Forward pass
    trainer.net.eval()
    tensor = forward(trainer, states, steps)

    if tensor.dim() == 3:
        tensor = tensor[0]

    deltas = tensor.detach().cpu().numpy().transpose(1, 0).astype(np.float32)

    deformed_points = states.detach().cpu().numpy()

    if deformed_points.ndim == 3:
        deformed_points = deformed_points[0]

    if deformed_points.shape[0] == 3:
        deformed_points = deformed_points.T

    # Surface reconstruction
    result = splashsurf.reconstruct_surface(
        deformed_points.astype(np.float32),
        particle_radius=0.05,
        smoothing_length=5,
        cube_size=0.5
    )

    triangles = result.mesh.triangles
    triangles[:, [1, 2]] = triangles[:, [2, 1]]  # invert normals

    vertices = result.mesh.vertices

    is_pressing = False
    return vertices, triangles, is_pressing


def handle_strike(req: ClientRequest) -> None:
    return


def handle_heat(req: ClientRequest) -> None:
    return


def handle_reset(req: ClientRequest) -> None:
    return


def handle_undo(req: ClientRequest) -> None:
    return


# ---------------------------
# Binary protocol packing
# ---------------------------

def make_binary_reply(vertices: np.ndarray,
                      faces: np.ndarray,
                      is_pressing: bool) -> bytes:

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
# WebSocket server
# ---------------------------

async def client_handler(ws: WebSocketServerProtocol, trainer: Trainer):
    print(f"[connect] {ws.remote_address}")

    try:
        async for message in ws:
            if isinstance(message, bytes):
                print("[warn] unexpected binary message")
                continue

            req = ClientRequest.from_json(message)

            if req.request == "update":
                vertices, faces, is_pressing = handle_update(req, trainer)

            elif req.request == "strike":
                handle_strike(req)
                vertices, faces, is_pressing = handle_update(req, trainer)

            elif req.request == "heat":
                handle_heat(req)
                vertices, faces, is_pressing = handle_update(req, trainer)

            elif req.request == "reset":
                handle_reset(req)
                vertices, faces, is_pressing = handle_update(req, trainer)

            elif req.request == "undo":
                handle_undo(req)
                vertices, faces, is_pressing = handle_update(req, trainer)

            else:
                print(f"[warn] unknown request: {req.request}")
                vertices, faces, is_pressing = handle_update(req, trainer)

            reply = make_binary_reply(vertices, faces, is_pressing)
            await ws.send(reply)

    except websockets.ConnectionClosed:
        pass
    except Exception as e:
        print(f"[error] {e}")
    finally:
        print(f"[disconnect] {ws.remote_address}")


# ---------------------------
# Main
# ---------------------------

async def main(host: str = "localhost", port: int = 8765):

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