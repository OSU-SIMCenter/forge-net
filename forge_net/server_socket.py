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
from typing import List, Tuple, Optional

import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

from pathlib import Path
import torch
from forge_net.utils.utils import *

from model.trainer import Trainer
from utils.utils import * 
import yaml
import os

from eval import evaluate, forward
from main import make_dataloaders

import pysplashsurf as splashsurf


# ---------------------------
# Data model (incoming)
# ---------------------------

@dataclass
class ClientRequest:
    request: str
    vertices: np.ndarray
    translation: float
    rotation: float
    force: float

    @staticmethod
    def from_json(s: str) -> "ClientRequest":
        obj = json.loads(s)

        raw_vertices = obj.get("vertices", [])

        vertices = np.asarray(raw_vertices, dtype=np.float32)

        return ClientRequest(
            request=str(obj.get("request", "")),
            vertices=vertices,
            translation=float(obj.get("translation", 0.0)),
            rotation=float(obj.get("rotation", 0.0)),
            force=float(obj.get("force", 0.0)),
        )
    

    # utils.py barycentric sampling

    # 


# ---------------------------
# Application-specific stubs
# ---------------------------

def handle_update(req: ClientRequest) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Return (vertices_f32_flat, faces_i32_flat, is_pressing).
    vertices: flattened float32 array length = N*3
    faces: flattened int32 array length = M*3
    is_pressing: bool

    App-specific: fill these in.
    """
    # Example dummy mesh: a single triangle
    # vertices = np.array(
    #     [0.0, 0.0, 0.0,
    #      1.0, 0.0, 0.0,
    #      0.0, 1.0, 0.0,
    #      1.0, 1.0, 0.0],
    #     dtype=np.float32
    # )


    # most of this should happen in initialization instead of every strike
    base_path = get_project_root()
    run_name = "mse_1024_unmaksed_seeded_w_tri_ids"
    config_path = base_path / "runs" / run_name / "config_out.yml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    trainer = Trainer(config, log_to_tb=False)
    # evaluate(config, trainer)
    batch_size = 2  # Set batch size to 2 or more
    states = np.ones((batch_size, 3, 1000))  # Shape: (B, C, N)
    states = torch.tensor(states, dtype=torch.float32)

    action_dims = config["network"]["action_dims"]
    steps = np.ones((batch_size, action_dims))  # Shape: (B, action_dims)
    steps = torch.tensor(steps, dtype=torch.float32)
    tensor = forward(trainer, states, steps)


    if tensor.dim() == 3:
        tensor = tensor[0]
    vertices = tensor.detach().cpu().numpy()
    vertices.astype(np.float32).reshape(-1)


    faces = np.array(np.random.randint(0, 100, size=(1, 600)), dtype=np.int32)

    result = splashsurf.reconstruct_surface(
        vertices,
        particle_radius=0.5,
        smoothing_length=2,
        cube_size=0.5
    )

    is_pressing = False
    return result.mesh.vertices, result.mesh.triangles, is_pressing


def handle_strike(req: ClientRequest) -> None:
    """App-specific stub."""
    return


def handle_heat(req: ClientRequest) -> None:
    """App-specific stub."""
    return


def handle_reset(req: ClientRequest) -> None:
    """App-specific stub."""
    return


def handle_undo(req: ClientRequest) -> None:
    """App-specific stub."""
    return


# ---------------------------
# Binary protocol packing
# ---------------------------

def make_binary_reply(vertices_f32_flat: np.ndarray,
                      faces_i32_flat: np.ndarray,
                      is_pressing: bool) -> bytes:
    """
    Pack bytes that Unity's UpdateBillet(byte[]) can parse.
    """
    # Ensure correct dtypes + contiguous memory
    v = np.ascontiguousarray(vertices_f32_flat, dtype=np.float32)
    f = np.ascontiguousarray(faces_i32_flat, dtype=np.int32)

    header_obj = {
        "is_pressing": bool(is_pressing),
        "counts": {
            "vertices": int(v.size),  # flattened count
            "faces": int(f.size),     # flattened count
        },
    }
    header_json = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")

    # Little-endian uint32 length prefix
    prefix = struct.pack("<I", len(header_json))

    # Body: raw little-endian float32 and int32
    body = v.tobytes(order="C") + f.tobytes(order="C")

    return prefix + header_json + body


# ---------------------------
# WebSocket server handler
# ---------------------------

async def client_handler(ws: WebSocketServerProtocol):
    print(f"[connect] {ws.remote_address}")
    try:
        async for message in ws:
            # Unity uses SendText, so expect str. If bytes arrive, ignore or handle.
            if isinstance(message, bytes):
                print("[warn] received unexpected binary message from client; ignoring")
                continue

            req = ClientRequest.from_json(message)

            # Route request
            if req.request == "update":
                vertices, faces, is_pressing = handle_update(req)
                reply = make_binary_reply(vertices, faces, is_pressing)
                await ws.send(reply)

            elif req.request == "strike":
                handle_strike(req)
                # Optionally still send a mesh update:
                vertices, faces, is_pressing = handle_update(req)
                await ws.send(make_binary_reply(vertices, faces, is_pressing))
                print("returned strike packet")

            elif req.request == "heat":
                handle_heat(req)
                vertices, faces, is_pressing = handle_update(req)
                await ws.send(make_binary_reply(vertices, faces, is_pressing))

            elif req.request == "reset":
                handle_reset(req)
                vertices, faces, is_pressing = handle_update(req)
                await ws.send(make_binary_reply(vertices, faces, is_pressing))

            elif req.request == "undo":
                handle_undo(req)
                vertices, faces, is_pressing = handle_update(req)
                await ws.send(make_binary_reply(vertices, faces, is_pressing))

            else:
                print(f"[warn] unknown request: {req.request!r}")
                # Safe default: send something valid so Unity doesn't choke
                vertices, faces, is_pressing = handle_update(req)
                await ws.send(make_binary_reply(vertices, faces, is_pressing))

    except websockets.ConnectionClosed:
        pass
    except Exception as e:
        print(f"[error] {e}")
    finally:
        print(f"[disconnect] {ws.remote_address}")


async def main(host: str = "localhost", port: int = 8765):

    base_path = get_project_root()
    run_name = "mse_1024_unmasked"
    config_path = base_path / "runs" / run_name / "config_out.yml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    # train_loader, test_loader = make_dataloaders(config)
    # trainer = Trainer(config, train_loader, log_to_tb=False)
    # # evaluate(config, trainer)
    # evaluate_series(config, trainer)
    print("done!")


    print(f"Starting server on ws://{host}:{port}")
    async with websockets.serve(client_handler, host, port, max_size=None):
        await asyncio.Future()  # run forever


if __name__ == "__main__":

    print("done!")

    asyncio.run(main())
