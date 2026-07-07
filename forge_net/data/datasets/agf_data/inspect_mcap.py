"""
Temp script to inspect the MCAP dataset and understand how to map it
to the same state-action tensor format as process_data.py (SQLite path).

SQLite strike schema (for reference):
  step_number | position (JSON [x,y,z]) | rotation (JSON [x,y,z,w])
  | result (JSON {Vertices, Triangles, Steps}) | series_id | press_id | score

process_data.py output keys:
  coords_t, coords_tp1   -- sampled mesh vertices before/after each strike
  steps                  -- sum of Steps from result JSON
  positions              -- press position [x,y,z]
  rotations              -- press rotation quaternion [x,y,z,w]
  series_ids, series_lengths, meshes, meshes_tp1, tri_ids, bary_coords

Run with:
  conda run -n jax_forgeRL python inspect_mcap.py
"""

import json
from collections import defaultdict
from pathlib import Path
from mcap.reader import make_reader

MCAP_PATH = Path(__file__).parent / "20260608_151857_target3.mcap"


# ─── 1. High-level structure ─────────────────────────────────────────────────

def inspect_structure(path: Path):
    print("=" * 60)
    print("FILE:", path.name, f"({path.stat().st_size / 1e6:.1f} MB)")
    print("=" * 60)

    with open(path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()

    if summary is None:
        print("No summary index — will need to scan messages.")
        return

    print("\n--- SCHEMAS ---")
    for sid, schema in summary.schemas.items():
        print(f"  [{sid}] name={schema.name!r}  encoding={schema.encoding!r}")

    print("\n--- CHANNELS ---")
    for cid, ch in summary.channels.items():
        schema_name = summary.schemas[ch.schema_id].name if ch.schema_id in summary.schemas else "?"
        print(f"  [{cid}] topic={ch.topic!r}  schema={schema_name!r}  "
              f"encoding={ch.message_encoding!r}")

    print("\n--- MESSAGE COUNTS PER CHANNEL ---")
    if summary.statistics:
        stats = summary.statistics
        print(f"  Total messages : {stats.message_count}")
        print(f"  Time range     : {stats.message_start_time} – {stats.message_end_time}")
        for cid, count in stats.channel_message_counts.items():
            topic = summary.channels[cid].topic
            print(f"  {topic!r:40s}  {count:>8,} msgs")


# ─── 2. Sample messages per topic ────────────────────────────────────────────

def sample_messages(path: Path, n_per_topic: int = 3):
    print("\n" + "=" * 60)
    print("SAMPLE MESSAGES (first N per topic)")
    print("=" * 60)

    seen = defaultdict(int)
    done = set()

    with open(path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        n_topics = len(summary.channels) if summary else 99

        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            if seen[topic] >= n_per_topic:
                done.add(topic)
                if len(done) == n_topics:
                    break
                continue

            seen[topic] += 1
            idx = seen[topic]

            print(f"\n[{topic}] msg #{idx}  t={message.log_time}")
            print(f"  encoding: {channel.message_encoding}")

            raw = message.data
            # Try to decode as JSON
            if channel.message_encoding in ("json", ""):
                try:
                    decoded = json.loads(raw)
                    _print_json_structure(decoded, indent=4)
                    continue
                except Exception:
                    pass
            # Fallback: show raw bytes summary
            print(f"  raw bytes ({len(raw)}): {raw[:120]}...")


def _print_json_structure(obj, indent=4, depth=0, max_depth=3):
    pad = " " * indent * depth
    if depth > max_depth:
        print(pad + "...")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print(f"{pad}{k}: {type(v).__name__}({_shape(v)})")
                _print_json_structure(v, indent, depth + 1, max_depth)
            else:
                print(f"{pad}{k}: {v!r}")
    elif isinstance(obj, list):
        if len(obj) == 0:
            print(pad + "(empty list)")
        elif isinstance(obj[0], (int, float)):
            arr_info = f"list[{len(obj)}] numeric, first={obj[0]:.4g}, last={obj[-1]:.4g}"
            print(pad + arr_info)
        elif isinstance(obj[0], dict):
            print(pad + f"list[{len(obj)}] of dicts, first item keys: {list(obj[0].keys())}")
            _print_json_structure(obj[0], indent, depth + 1, max_depth)
        else:
            print(pad + f"list[{len(obj)}]: {obj[:3]!r} ...")
    else:
        print(pad + repr(obj))


def _shape(obj):
    if isinstance(obj, list):
        return f"len={len(obj)}"
    if isinstance(obj, dict):
        return f"keys={list(obj.keys())[:6]}"
    return ""


# ─── 3. Per-topic message counts (fallback if no summary stats) ──────────────

def count_messages_by_topic(path: Path):
    print("\n" + "=" * 60)
    print("COUNTING MESSAGES BY TOPIC (full scan)")
    print("=" * 60)
    counts = defaultdict(int)
    with open(path, "rb") as f:
        reader = make_reader(f)
        for _, channel, _ in reader.iter_messages():
            counts[channel.topic] += 1
    for topic, count in sorted(counts.items()):
        print(f"  {topic!r:50s}  {count:>8,}")


# ─── 4. SQLite reference snapshot ────────────────────────────────────────────

def inspect_sqlite(db_path: str):
    import sqlite3
    print("\n" + "=" * 60)
    print("SQLITE REFERENCE  (safe_stock_cogging2.db)")
    print("=" * 60)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(DISTINCT series_id) FROM strike")
    n_series = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM strike")
    n_strikes = cur.fetchone()[0]
    print(f"  series: {n_series}   strikes: {n_strikes}")

    cur.execute("SELECT * FROM strike LIMIT 1")
    row = cur.fetchone()
    cols = [d[0] for d in cur.description]
    print("\n  Strike row sample:")
    for col, val in zip(cols, row):
        if isinstance(val, str) and len(val) > 120:
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict):
                    info = {k: (f"list[{len(v)}]" if isinstance(v, list) else v)
                            for k, v in parsed.items()}
                    print(f"    {col}: JSON {info}")
                else:
                    print(f"    {col}: JSON list[{len(parsed)}]")
            except Exception:
                print(f"    {col}: str[{len(val)}]")
        else:
            print(f"    {col}: {val!r}")
    conn.close()


# ─── 5. Count press cycles (idle→active→idle transitions) ────────────────────

def count_press_cycles(path: Path):
    print("\n" + "=" * 60)
    print("PRESS CYCLE DETECTION")
    print("=" * 60)

    cycle_events = []   # (t_start, t_end, min_pos_mm, max_force_kn, max_stroke_mm, torm_pos)
    in_cycle = False
    cycle_start_t = None
    cycle_min_pos = None
    cycle_max_force = 0.0
    cycle_max_stroke = 0.0
    last_torm_pos = None

    with open(path, "rb") as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages():
            topic = channel.topic

            if topic == "hmr/torm/state":
                m = json.loads(message.data)
                last_torm_pos = m["actual_position"]

            if topic == "hmr/press/state":
                m = json.loads(message.data)
                is_idle = m["is_idle"]
                t = message.log_time

                if not in_cycle and not is_idle:
                    in_cycle = True
                    cycle_start_t = t
                    cycle_min_pos = m["live_position_mm"]
                    cycle_max_force = m["live_force_kn"]
                    cycle_max_stroke = m["live_stroke_mm"]

                elif in_cycle and not is_idle:
                    if m["live_position_mm"] < cycle_min_pos:
                        cycle_min_pos = m["live_position_mm"]
                    if m["live_force_kn"] > cycle_max_force:
                        cycle_max_force = m["live_force_kn"]
                    if m["live_stroke_mm"] > cycle_max_stroke:
                        cycle_max_stroke = m["live_stroke_mm"]

                elif in_cycle and is_idle:
                    in_cycle = False
                    cycle_events.append({
                        "t_start": cycle_start_t,
                        "t_end": t,
                        "duration_ms": (t - cycle_start_t) / 1e6,
                        "min_pos_mm": cycle_min_pos,
                        "max_force_kn": cycle_max_force,
                        "max_stroke_mm": cycle_max_stroke,
                        "torm_pos_xyz": last_torm_pos[:3] if last_torm_pos else None,
                    })

    print(f"  Total press cycles detected: {len(cycle_events)}")
    print(f"\n  First 10 cycles:")
    print(f"  {'#':>4}  {'t_start':>22}  {'dur_ms':>8}  {'force_kn':>10}  {'stroke_mm':>10}  {'min_pos_mm':>12}  torm_xyz")
    for i, ev in enumerate(cycle_events[:10]):
        xyz = ev["torm_pos_xyz"]
        xyz_str = f"[{xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}]" if xyz else "N/A"
        print(f"  {i:>4}  {ev['t_start']:>22}  {ev['duration_ms']:>8.1f}  "
              f"{ev['max_force_kn']:>10.3f}  {ev['max_stroke_mm']:>10.2f}  "
              f"{ev['min_pos_mm']:>12.2f}  {xyz_str}")

    if len(cycle_events) > 1:
        gaps = [(cycle_events[i+1]["t_start"] - cycle_events[i]["t_end"]) / 1e9
                for i in range(len(cycle_events)-1)]
        print(f"\n  Inter-cycle gap stats (seconds):")
        import numpy as np
        gaps = np.array(gaps)
        print(f"    mean={gaps.mean():.2f}  median={np.median(gaps):.2f}  "
              f"min={gaps.min():.2f}  max={gaps.max():.2f}")
        unique_positions = set(tuple(round(x,1) for x in ev["torm_pos_xyz"][:2])
                               for ev in cycle_events if ev["torm_pos_xyz"])
        print(f"  Unique (X,Y) torm positions across cycles: {len(unique_positions)}")
        print(f"  Sample: {list(unique_positions)[:8]}")

    return cycle_events


# ─── 6. Conversion approach notes ────────────────────────────────────────────

APPROACH_NOTES = """
=== KEY FINDINGS ===

This MCAP is REAL HARDWARE data (not simulation). There are NO mesh vertices/triangles.
Topics and their roles:

  hmr/press/state   (~252k msgs, ~65 Hz)
    - live_force_kn, live_stroke_mm, live_position_mm  ← press kinematics
    - is_idle / cycle_end flags  ← episode boundary signal
    ACTION: press stroke depth (live_stroke_mm at cycle peak) or target position

  hmr/torm/state    (~84k msgs, ~21 Hz)
    - actual_position: 9-element array
        [0:3] = X, Y, Z TCP position in mm (e.g. [128.5, -90.26, 0.0])
        [3:9] = likely zero-padded / orientation placeholder (all ~0)
    - command: string (e.g. 'G0 X.. Y..')  ← robot move commands
    ACTION: (X, Y) robot TCP position = where on workpiece the press lands

  hmr/ard/state     (~136k msgs, ~35 Hz)
    - digital: limit switches, HLFB signals, door/estop state
    - analog.ram_linear_position: linear encoder (mm x scaling?)
    Supporting sensor, not primary state/action

  hmr/sensors/thermalcam  (~106k msgs, protobuf foxglove.RawImage)
    - 320x256 or similar thermal image of workpiece
    POSSIBLE STATE: temperature distribution = proxy for workpiece condition

=== PROPOSED CONVERSION APPROACH ===

Since there is no FEM mesh, state representation must differ from the SQLite pipeline.
Two realistic options:

OPTION A — Kinematics only (closest to SQLite actions, no mesh state)
  State_t  = [torm_x, torm_y, torm_z, press_pos_mm, press_force_kn]  at cycle start
  Action   = delta_torm_xy + press_stroke_mm  (what changed between cycles)
  State_t+1 = same fields after the move
  Episode  = grouped by inter-cycle gap threshold (seconds)
  Output   = same npz keys: coords_t/coords_tp1 replaced by state_t/state_tp1,
             positions/rotations hold torm_xyz, steps holds press stroke depth

OPTION B — Thermal image as state (richer but heavier)
  State_t  = thermal image frame nearest to cycle start
  Action   = (delta_torm_xy, stroke_depth)
  State_t+1 = thermal image after cycle completes
  Output   = image tensors (N, H, W) + action tensors (N, 3)

CONCRETE PIPELINE (Option A):
  1. Scan MCAP, detect press cycles via is_idle F→T transitions
  2. For each cycle, record:
       torm_pos_xyz  (from nearest hmr/torm/state message by timestamp)
       stroke_depth  (max live_stroke_mm during cycle)
       peak_force    (max live_force_kn during cycle)
       t_start, t_end
  3. Group cycles into episodes using inter-cycle gap > threshold seconds
  4. For each consecutive cycle pair (t, t+1) within an episode:
       state_t   = torm_xyz_t  + [stroke_t, force_t]
       state_tp1 = torm_xyz_tp1 + [stroke_tp1, force_tp1]
       action    = torm_xyz_tp1 - torm_xyz_t (delta move) + [stroke_tp1]
  5. Save npz with same layout as process_data.py output
     (coords_t → state_t, coords_tp1 → state_tp1, steps → stroke_depth)

  Key open question: what defines an episode boundary?
  Run count_press_cycles() to see inter-cycle gap distribution and pick threshold.
"""


if __name__ == "__main__":
    inspect_structure(MCAP_PATH)
    sample_messages(MCAP_PATH, n_per_topic=2)
    inspect_sqlite("/local/scratch/groves/jax-forgeRL/data/rand_cogging/safe_stock_cogging2.db")
    cycles = count_press_cycles(MCAP_PATH)
    print(APPROACH_NOTES)
