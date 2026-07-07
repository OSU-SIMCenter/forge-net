"""
Parse foxglove.RawImage protobuf frames from hmr/sensors/thermalcam manually
(no mcap_protobuf decoder needed).

Run: conda run -n jax_forgeRL python inspect_thermal.py
"""

import struct
import numpy as np
from pathlib import Path
from mcap.reader import make_reader
from PIL import Image

MCAP_PATH = "20260608_151857_target3.mcap"
OUT_DIR = Path("thermal_frames")
OUT_DIR.mkdir(exist_ok=True)


def parse_varint(data: bytes, pos: int):
    result, shift = 0, 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def parse_raw_image(data: bytes) -> dict:
    """Minimal protobuf parser for foxglove.RawImage.
    Field map (from Foxglove schema):
      1 = timestamp (LEN, nested)
      2 = frame_id  (LEN, string) -- often absent
      3 = width     (fixed32 or varint)
      4 = height    (fixed32 or varint)
      5 = encoding  (LEN, string)
      6 = step      (varint)
      7 = data      (LEN, bytes)
    """
    pos = 0
    out = {}
    while pos < len(data):
        tag, pos = parse_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x7
        if wire == 0:    # varint
            v, pos = parse_varint(data, pos)
            out[field] = v
        elif wire == 2:  # length-delimited
            length, pos = parse_varint(data, pos)
            out[field] = data[pos:pos + length]
            pos += length
        elif wire == 5:  # 32-bit fixed
            out[field] = struct.unpack_from('<I', data, pos)[0]
            pos += 4
        else:
            print(f"  Warning: unknown wire type {wire} at pos {pos}, stopping parse")
            break
    return out


def extract_frame(raw_msg_data: bytes):
    parsed = parse_raw_image(raw_msg_data)
    # This camera's schema omits frame_id so field numbers are shifted:
    #   1=timestamp  2=width(fixed32)  3=height(fixed32)
    #   4=encoding(str)  5=step(fixed32)  6=data(bytes)
    enc_field = parsed.get(4, b'')
    if isinstance(enc_field, (bytes, bytearray)):
        enc = enc_field.decode('utf-8', errors='replace')
    else:
        enc = str(enc_field)

    width  = parsed.get(2, 0)
    height = parsed.get(3, 0)
    step   = parsed.get(5, 0)
    raw    = parsed.get(6, b'')

    return enc, width, height, step, raw, parsed


def to_uint8(img16: np.ndarray) -> np.ndarray:
    lo, hi = img16.min(), img16.max()
    if hi == lo:
        return np.zeros_like(img16, dtype=np.uint8)
    return ((img16.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)


if __name__ == "__main__":
    # ── 1. Parse the first frame, print structure ──────────────────────────
    print("=" * 60)
    print("THERMAL FRAME STRUCTURE")
    print("=" * 60)

    first_frame = None
    first_ts = None

    with open(MCAP_PATH, "rb") as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages(topics=["hmr/sensors/thermalcam"]):
            enc, w, h, step, raw, parsed = extract_frame(message.data)
            print(f"  encoding : {enc!r}")
            print(f"  width    : {w}")
            print(f"  height   : {h}")
            print(f"  step     : {step}  (bytes per row, expected {w*2} for 16UC1)")
            print(f"  data len : {len(raw)}  (expected {w*h*2} = {w*h*2})")
            print(f"  all fields present: {list(parsed.keys())}")
            first_frame = (enc, w, h, raw)
            first_ts = message.log_time
            break

    enc, w, h, raw = first_frame
    if enc == "16UC1" and len(raw) == w * h * 2:
        img16 = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
        print(f"\n  pixel stats: min={img16.min()}  max={img16.max()}  "
              f"mean={img16.mean():.1f}  std={img16.std():.1f}")
        img8 = to_uint8(img16)
        out_path = OUT_DIR / "frame_0000.png"
        Image.fromarray(img8).save(out_path)
        print(f"\n  Saved normalized frame → {out_path}")
    else:
        print(f"  WARNING: unexpected encoding or data length, got enc={enc!r} len={len(raw)}")

    # ── 2. Sample frames spanning the full session ─────────────────────────
    print("\n" + "=" * 60)
    print("SAMPLING 20 FRAMES ACROSS SESSION")
    print("=" * 60)

    summary_ts_end = 1780950259773829700
    summary_ts_start = 1780946337927661100
    total_span = summary_ts_end - summary_ts_start

    target_ts = [summary_ts_start + int(total_span * i / 19) for i in range(20)]
    saved = []

    with open(MCAP_PATH, "rb") as f:
        reader = make_reader(f)
        tgt_idx = 0
        for _, channel, message in reader.iter_messages(topics=["hmr/sensors/thermalcam"]):
            if tgt_idx >= len(target_ts):
                break
            if message.log_time >= target_ts[tgt_idx]:
                enc, w, h, step, raw, _ = extract_frame(message.data)
                if enc == "16UC1" and len(raw) == w * h * 2:
                    img16 = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
                    img8 = to_uint8(img16)
                    fname = OUT_DIR / f"frame_{tgt_idx:04d}.png"
                    Image.fromarray(img8).save(fname)
                    saved.append((tgt_idx, message.log_time, img16.min(), img16.max(), img16.mean()))
                tgt_idx += 1

    print(f"  Saved {len(saved)} frames to {OUT_DIR}/")
    print(f"  {'idx':>4}  {'log_time':>22}  {'min':>7}  {'max':>7}  {'mean':>8}")
    for idx, ts, mn, mx, mean in saved:
        print(f"  {idx:>4}  {ts:>22}  {mn:>7}  {mx:>7}  {mean:>8.1f}")

    # ── 3. Grab frames immediately before a known press cycle start ─────────
    print("\n" + "=" * 60)
    print("FRAMES AROUND A PRESS CYCLE")
    print("=" * 60)

    import json
    from collections import deque

    # Find the first real press cycle (force > 10 kN)
    first_cycle_t = None
    with open(MCAP_PATH, "rb") as f:
        reader = make_reader(f)
        prev_idle = True
        for _, channel, message in reader.iter_messages(topics=["hmr/press/state"]):
            m = json.loads(message.data)
            idle = m["is_idle"]
            if prev_idle and not idle and m["live_force_kn"] > 10:
                first_cycle_t = message.log_time
                break
            prev_idle = idle

    print(f"  First substantial press cycle at t={first_cycle_t}")
    if first_cycle_t is None:
        print("  No cycle found.")
        exit()

    # Use the 10th substantial cycle so thermal cam is definitely running
    CYCLE_TARGET = 10
    with open(MCAP_PATH, "rb") as f:
        reader = make_reader(f)
        prev_idle = True
        count = 0
        for _, channel, message in reader.iter_messages(topics=["hmr/press/state"]):
            m = json.loads(message.data)
            idle = m["is_idle"]
            if prev_idle and not idle and m["live_force_kn"] > 10:
                if count == CYCLE_TARGET:
                    first_cycle_t = message.log_time
                    break
                count += 1
            prev_idle = idle

    print(f"  Using cycle #{CYCLE_TARGET} at t={first_cycle_t}")

    # Collect thermal frames in [-5s, +5s] window around cycle start
    window_start = first_cycle_t - 5_000_000_000   # 5s before
    window_end   = first_cycle_t + 5_000_000_000   # 5s after

    cycle_frames = []
    with open(MCAP_PATH, "rb") as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages(
                topics=["hmr/sensors/thermalcam"],
                start_time=window_start, end_time=window_end):
            enc, w, h, step, raw, _ = extract_frame(message.data)
            if enc == "16UC1" and len(raw) == w * h * 2:
                img16 = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
                cycle_frames.append((message.log_time, img16))

    print(f"  Frames in ±5s window: {len(cycle_frames)}")
    if cycle_frames:
        pre = [(t, f) for t, f in cycle_frames if t < first_cycle_t]
        post = [(t, f) for t, f in cycle_frames if t >= first_cycle_t]
        print(f"  Pre-strike: {len(pre)}  Post-strike: {len(post)}")

        # Save a few pre-strike frames
        for i, (ts, img16) in enumerate(pre[-5:]):
            dt_ms = (ts - first_cycle_t) / 1e6
            img8 = to_uint8(img16)
            fname = OUT_DIR / f"prestrike_{i:02d}_{dt_ms:+.0f}ms.png"
            Image.fromarray(img8).save(fname)
            print(f"    Saved {fname.name}  min={img16.min()}  max={img16.max()}  mean={img16.mean():.1f}")

        # Pixel-wise stats across pre-strike frames (background stability)
        if len(pre) > 3:
            stack = np.stack([f for _, f in pre], axis=0).astype(np.float32)
            temporal_std = stack.std(axis=0)
            print(f"\n  Pre-strike temporal std across pixels:")
            print(f"    mean_std={temporal_std.mean():.2f}  max_std={temporal_std.max():.2f}")
            print(f"    High-variance pixels (std>100): {(temporal_std > 100).sum()}")
            print(f"    -> Hot workpiece region will show high std if it's moving/deforming")
