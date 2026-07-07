"""
Thermal-to-point-cloud reconstruction for forged billet.

Approach:
  - 3 A-axis angles (0°, 4°, 16°) give 3 silhouette width measurements per cross-section
  - Fit ellipse W(θ) = 2√(a²sin²θ + b²cos²θ) at each longitudinal Y slice
  - Sample fitted ellipse surface → 3D point cloud
  - Compare reconstructed dimensions across strikes to track deformation

Run: conda run -n jax_forgeRL python thermal_reconstruction.py
"""

import json
import re
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from mcap.reader import make_reader
from PIL import Image

MCAP_PATH = "20260608_151857_target3.mcap"
OUT_DIR = Path("reconstruction_out")
OUT_DIR.mkdir(exist_ok=True)

A_RE = re.compile(r'A([\-\d.]+)')
THERMAL_START_T = 1780946355875537500   # first thermal frame


# ─── protobuf helpers (reused from inspect_thermal.py) ───────────────────────

def parse_varint(data, pos):
    result, shift = 0, 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80): break
        shift += 7
    return result, pos

def parse_raw_image(data):
    pos = 0; out = {}
    while pos < len(data):
        tag, pos = parse_varint(data, pos)
        field, wire = tag >> 3, tag & 0x7
        if wire == 0:
            v, pos = parse_varint(data, pos); out[field] = v
        elif wire == 2:
            l, pos = parse_varint(data, pos); out[field] = data[pos:pos+l]; pos += l
        elif wire == 5:
            out[field] = struct.unpack_from('<I', data, pos)[0]; pos += 4
        else:
            break
    return out

def decode_thermal(msg_data):
    p = parse_raw_image(msg_data)
    enc = p.get(4, b'').decode() if isinstance(p.get(4), bytes) else ''
    w, h, raw = p.get(2, 0), p.get(3, 0), p.get(6, b'')
    if enc == '16UC1' and len(raw) == w * h * 2:
        return np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
    return None


# ─── 1. Build cycle table ─────────────────────────────────────────────────────

def build_cycle_table(mcap_path):
    """
    Scan the full MCAP once, returning a list of dicts, one per press cycle:
      t_start, t_end, peak_force_kn, max_stroke_mm, torm_y, torm_a, torm_x
    """
    print("Scanning press cycles and torm state...")
    cycles = []

    last_torm = {'x': None, 'y': None, 'a': None, 't': None}
    in_cycle = False
    cycle_start = None
    peak_force = 0.0
    max_stroke = 0.0

    with open(mcap_path, 'rb') as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages(
                topics=['hmr/press/state', 'hmr/torm/state']):

            if channel.topic == 'hmr/torm/state':
                m = json.loads(message.data)
                pos = m['actual_position']
                last_torm['x'] = pos[0]
                last_torm['y'] = pos[1]
                last_torm['t'] = message.log_time
                cmd = m.get('command', '')
                if cmd:
                    ma = A_RE.search(cmd)
                    if ma:
                        last_torm['a'] = float(ma.group(1))

            elif channel.topic == 'hmr/press/state':
                m = json.loads(message.data)
                idle = m['is_idle']
                t = message.log_time

                if not in_cycle and not idle:
                    in_cycle = True
                    cycle_start = t
                    peak_force = m['live_force_kn']
                    max_stroke = m['live_stroke_mm']

                elif in_cycle and not idle:
                    peak_force = max(peak_force, m['live_force_kn'])
                    max_stroke = max(max_stroke, m['live_stroke_mm'])

                elif in_cycle and idle:
                    in_cycle = False
                    if (cycle_start > THERMAL_START_T
                            and peak_force > 20.0
                            and last_torm['y'] is not None
                            and last_torm['a'] is not None):
                        cycles.append({
                            't_start': cycle_start,
                            't_end': t,
                            'peak_force_kn': peak_force,
                            'max_stroke_mm': max_stroke,
                            'torm_x': last_torm['x'],
                            'torm_y': last_torm['y'],
                            'torm_a': last_torm['a'],
                        })

    print(f"  Found {len(cycles)} usable cycles (>20kN, after thermal start)")
    return cycles


# ─── 2. Extract silhouette from thermal frame ─────────────────────────────────

def get_silhouette(img16, percentile=98.0, min_aspect=1.5, min_area=50):
    """
    Find the hot workpiece billet using percentile threshold + connected-component
    filtering. Robust to thermal drift of background and press die.

    Strategy:
      1. Threshold at the given percentile (top 2% of pixels by default)
      2. Label connected components
      3. Keep the largest component that is taller than it is wide (aspect ≥ min_aspect)
         and has at least min_area pixels

    Returns:
      mask          : bool (H,W) — True = billet pixels
      width_profile : (H,) — billet pixel width per row (0 outside billet rows)
      col_centroid  : (H,) — column centroid per row (nan outside billet rows)
      bbox          : (row_min, row_max, col_min, col_max) or None
    """
    from scipy.ndimage import label

    flat = img16.astype(np.float32)
    thresh = np.percentile(flat, percentile)
    hot = flat >= thresh

    # Label connected components
    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
    labeled, n_comp = label(hot, structure=structure)

    best_mask = None
    best_score = -1

    for comp_id in range(1, n_comp + 1):
        comp = labeled == comp_id
        area = comp.sum()
        if area < min_area:
            continue
        rows = np.where(comp.any(axis=1))[0]
        cols = np.where(comp.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            continue
        h = rows[-1] - rows[0] + 1
        w = cols[-1] - cols[0] + 1
        aspect = h / max(w, 1)
        if aspect < min_aspect:
            continue
        # Score: prefer tall+hot regions
        score = area * aspect
        if score > best_score:
            best_score = score
            best_mask = comp

    if best_mask is None:
        empty = np.zeros(img16.shape[0], dtype=np.float32)
        return np.zeros_like(hot), empty, np.full(img16.shape[0], np.nan), None

    rows_with_hot = best_mask.any(axis=1)
    row_min = int(np.where(rows_with_hot)[0][0])
    row_max = int(np.where(rows_with_hot)[0][-1])
    cols_with_hot = best_mask.any(axis=0)
    col_min = int(np.where(cols_with_hot)[0][0])
    col_max = int(np.where(cols_with_hot)[0][-1])

    width_profile = best_mask.sum(axis=1).astype(np.float32)
    col_centroid = np.where(
        best_mask.any(axis=1),
        (best_mask * np.arange(best_mask.shape[1])[None, :]).sum(axis=1) /
        np.maximum(best_mask.sum(axis=1), 1),
        np.nan,
    )

    return best_mask, width_profile, col_centroid, (row_min, row_max, col_min, col_max)


# ─── 3. Match nearest thermal frame to each cycle ────────────────────────────

def extract_silhouettes_for_cycles(mcap_path, cycles):
    """
    For each cycle, find the thermal frame closest to (but before) t_start.
    Extract silhouette metrics.
    Returns list parallel to cycles with silhouette data added.
    """
    print("Extracting silhouettes for each cycle...")

    # Build a sorted list of cycle start times for fast lookup
    cycle_starts = np.array([c['t_start'] for c in cycles])

    # Buffer for recent thermal frames (keep rolling window)
    frame_buffer = {}   # t -> img16

    results = [None] * len(cycles)
    matched = [False] * len(cycles)

    LOOKBACK_NS = 2_000_000_000  # 2 seconds lookback

    with open(mcap_path, 'rb') as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages(
                topics=['hmr/sensors/thermalcam']):
            t = message.log_time
            img16 = decode_thermal(message.data)
            if img16 is None:
                continue

            # Add to buffer
            frame_buffer[t] = img16

            # Trim old frames
            old_keys = [k for k in frame_buffer if k < t - LOOKBACK_NS]
            for k in old_keys:
                del frame_buffer[k]

            # Check which cycles this frame could serve (frame is just before t_start)
            for i, cyc in enumerate(cycles):
                if matched[i]:
                    continue
                # This frame is within [t_start-2s, t_start]
                if cyc['t_start'] - LOOKBACK_NS <= t <= cyc['t_start']:
                    # Keep updating — will end up with the most recent pre-strike frame
                    mask, width_profile, col_centroid, bbox = get_silhouette(img16)
                    results[i] = {
                        'frame_t': t,
                        'img16': img16,
                        'mask': mask,
                        'width_profile': width_profile,
                        'col_centroid': col_centroid,
                        'bbox': bbox,
                    }
                elif t > cyc['t_start'] and results[i] is not None:
                    matched[i] = True

    n_matched = sum(r is not None for r in results)
    print(f"  Matched {n_matched}/{len(cycles)} cycles with a pre-strike thermal frame")
    return results


# ─── 4. Ellipse width model ───────────────────────────────────────────────────

def ellipse_projected_width(theta_deg, a, b):
    """
    Projected width of an ellipse with semi-axes a, b when rotated by theta_deg
    around the viewing direction.
    W(θ) = 2 * sqrt(a² sin²θ + b² cos²θ)
    Here a = semi-axis ⊥ to default view, b = semi-axis ∥ to default view.
    """
    theta = np.deg2rad(theta_deg)
    return 2.0 * np.sqrt(a**2 * np.sin(theta)**2 + b**2 * np.cos(theta)**2)


# ─── 5. Per-Y-slice ellipse fitting ──────────────────────────────────────────

def fit_cross_sections(cycles, silhouettes, px_per_mm_y=None, px_per_mm_x=None):
    """
    Group cycles by Y position and A angle.
    For each unique Y position, fit ellipse to width measurements at 3 A values.

    Returns dict: y_mm -> {'a_mm', 'b_mm', 'residual', 'n_samples', 'widths_px'}
    """
    # Round Y to 1mm and A to nearest degree for grouping
    from collections import defaultdict
    groups = defaultdict(lambda: defaultdict(list))   # groups[y_rounded][a] = [width_px, ...]

    for cyc, sil in zip(cycles, silhouettes):
        if sil is None or sil['bbox'] is None:
            continue
        y_key = round(cyc['torm_y'])
        a_key = round(cyc['torm_a'])
        wp = sil['width_profile']
        if wp is None:
            continue
        # Use the median width of the hot rows as the representative silhouette width
        bbox = sil['bbox']
        hot_rows = wp[bbox[0]:bbox[1]+1]
        if hot_rows.max() < 5:   # skip frames with tiny/noisy detections
            continue
        median_w = float(np.median(hot_rows[hot_rows > 0]))
        groups[y_key][a_key].append(median_w)

    print(f"\nUnique Y positions with data: {len(groups)}")
    unique_a = set()
    for gd in groups.values():
        unique_a.update(gd.keys())
    print(f"A angles seen: {sorted(unique_a)}")

    # Fit ellipse per Y position
    results = {}
    A_VALS = sorted(unique_a)

    for y_key in sorted(groups.keys()):
        gd = groups[y_key]
        a_vals = [a for a in A_VALS if a in gd and len(gd[a]) >= 1]
        if len(a_vals) < 2:
            continue

        theta_arr = np.array([float(a) for a in a_vals])
        w_arr = np.array([np.median(gd[a]) for a in a_vals])

        try:
            # Initial guess: both semi-axes = half the median width
            w0 = w_arr.mean() / 2
            popt, pcov = curve_fit(
                ellipse_projected_width,
                theta_arr, w_arr,
                p0=[w0, w0],
                bounds=(1, np.inf),
                maxfev=2000
            )
            a_fit, b_fit = popt
            residual = np.sqrt(np.mean((ellipse_projected_width(theta_arr, *popt) - w_arr)**2))

            results[y_key] = {
                'a_px': a_fit,
                'b_px': b_fit,
                'residual_px': residual,
                'n_a_angles': len(a_vals),
                'widths_px': dict(zip(a_vals, w_arr.tolist())),
                'theta_arr': theta_arr,
                'w_arr': w_arr,
            }
        except Exception as e:
            pass

    print(f"Fitted cross-sections at {len(results)} Y positions")
    return results, groups


# ─── 6. Point cloud from ellipse fits ────────────────────────────────────────

def build_point_cloud(fit_results, px_per_mm=None, n_pts_per_slice=64):
    """
    For each Y slice with a fitted ellipse, sample n_pts_per_slice points
    around the ellipse circumference. Returns (N,3) array in pixel/mm units.

    Coordinate convention:
      Y_axis  = longitudinal (from torm Y, normalized to slice index here)
      XZ_plane = cross-section (ellipse with fitted semi-axes a,b)
    """
    pts = []
    y_keys = sorted(fit_results.keys())

    phi = np.linspace(0, 2 * np.pi, n_pts_per_slice, endpoint=False)

    for y_idx, y_key in enumerate(y_keys):
        r = fit_results[y_key]
        a, b = r['a_px'], r['b_px']
        x = a * np.cos(phi)
        z = b * np.sin(phi)
        y = np.full(n_pts_per_slice, float(y_key))
        pts.append(np.stack([x, y, z], axis=1))

    if not pts:
        return None
    cloud = np.concatenate(pts, axis=0)   # (N, 3)
    return cloud


# ─── 7. Visualise ────────────────────────────────────────────────────────────

def plot_cross_section_summary(fit_results, groups, out_dir):
    y_keys = sorted(fit_results.keys())
    if not y_keys:
        print("No fit results to plot.")
        return

    a_vals = [fit_results[y]['a_px'] for y in y_keys]
    b_vals = [fit_results[y]['b_px'] for y in y_keys]
    res    = [fit_results[y]['residual_px'] for y in y_keys]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(y_keys, a_vals, 'b.-', label='semi-axis a (px)')
    axes[0].plot(y_keys, b_vals, 'r.-', label='semi-axis b (px)')
    axes[0].set_xlabel('Y position (mm / torm units)')
    axes[0].set_ylabel('semi-axis (px)')
    axes[0].set_title('Fitted ellipse semi-axes vs Y position')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(y_keys, res, 'k.-')
    axes[1].set_xlabel('Y position')
    axes[1].set_ylabel('RMS residual (px)')
    axes[1].set_title('Ellipse fit residual')
    axes[1].grid(True)

    # 3D cross-section at a few Y slices
    ax3 = axes[2]
    sample_ys = y_keys[::max(1, len(y_keys)//8)]
    phi = np.linspace(0, 2*np.pi, 100)
    for y in sample_ys:
        a, b = fit_results[y]['a_px'], fit_results[y]['b_px']
        ax3.plot(a * np.cos(phi), b * np.sin(phi), alpha=0.6, label=f'Y={y}')
    ax3.set_aspect('equal')
    ax3.set_title('Cross-section ellipses (in pixels)')
    ax3.set_xlabel('a-axis (px)')
    ax3.set_ylabel('b-axis (px)')
    ax3.grid(True)

    plt.tight_layout()
    fig.savefig(out_dir / 'cross_sections.png', dpi=120)
    plt.close()
    print(f"Saved cross_sections.png")


def plot_point_cloud(cloud, out_dir):
    if cloud is None:
        return
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(cloud[::4, 0], cloud[::4, 1], cloud[::4, 2],
                s=1, c=cloud[::4, 2], cmap='plasma')
    ax1.set_title('3D point cloud (px units)')
    ax1.set_xlabel('X (a-axis)'); ax1.set_ylabel('Y (longitudinal)'); ax1.set_zlabel('Z (b-axis)')

    ax2 = fig.add_subplot(122)
    ax2.scatter(cloud[::4, 1], cloud[::4, 0], s=1, alpha=0.3, c='steelblue')
    ax2.set_title('Top view (Y vs X)')
    ax2.set_xlabel('Y (longitudinal)'); ax2.set_ylabel('X')
    ax2.set_aspect('equal'); ax2.grid(True)

    plt.tight_layout()
    fig.savefig(out_dir / 'point_cloud.png', dpi=120)
    plt.close()
    print("Saved point_cloud.png")


def plot_sample_silhouettes(cycles, silhouettes, out_dir, n=12):
    """Save a grid of sample silhouette images coloured by A angle."""
    a_colors = {0: 'cyan', 4: 'lime', 16: 'red'}
    samples = [(i, c, s) for i, (c, s) in enumerate(zip(cycles, silhouettes))
               if s is not None and s['bbox'] is not None][:n]

    if not samples:
        return

    fig, axes = plt.subplots(2, max(1, len(samples)//2), figsize=(20, 6))
    axes = axes.flatten()

    for ax, (i, cyc, sil) in zip(axes, samples):
        img = sil['img16'].astype(np.float32)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        ax.imshow(img, cmap='inferno', origin='upper')
        bbox = sil['bbox']
        from matplotlib.patches import Rectangle
        rect = Rectangle((bbox[2], bbox[0]), bbox[3]-bbox[2], bbox[1]-bbox[0],
                          linewidth=1.5, edgecolor=a_colors.get(round(cyc['torm_a']), 'white'),
                          facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f"Y={cyc['torm_y']:.1f} A={cyc['torm_a']:.0f}° F={cyc['peak_force_kn']:.0f}kN",
                     fontsize=7)
        ax.axis('off')

    for ax in axes[len(samples):]:
        ax.axis('off')

    plt.suptitle('Sample silhouettes (box colour: A=0°→cyan, 4°→lime, 16°→red)', fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / 'silhouette_grid.png', dpi=100)
    plt.close()
    print("Saved silhouette_grid.png")


def plot_width_vs_a(groups, out_dir):
    """For each Y position, plot measured width at each A angle."""
    y_keys = sorted(groups.keys())
    a_vals_all = sorted(set(a for gd in groups.values() for a in gd.keys()))

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(y_keys)-1, 1)) for i in range(len(y_keys))]

    for y_key, color in zip(y_keys, colors):
        gd = groups[y_key]
        xs, ys = [], []
        for a in sorted(gd.keys()):
            ws = gd[a]
            xs.append(float(a))
            ys.append(float(np.median(ws)))
        if len(xs) >= 2:
            ax.plot(xs, ys, '.-', color=color, alpha=0.7, linewidth=1)

    ax.set_xlabel('A angle (degrees)')
    ax.set_ylabel('Median silhouette width (px)')
    ax.set_title('Silhouette width vs A angle, coloured by Y position')
    ax.grid(True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(y_keys), max(y_keys)))
    plt.colorbar(sm, ax=ax, label='Y (torm mm)')
    fig.savefig(out_dir / 'width_vs_a.png', dpi=120)
    plt.close()
    print("Saved width_vs_a.png")


# ─── 8. Summary stats ────────────────────────────────────────────────────────

def print_summary(cycles, fit_results):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total usable cycles: {len(cycles)}")

    a_dist = {}
    for c in cycles:
        k = round(c['torm_a'])
        a_dist[k] = a_dist.get(k, 0) + 1
    print(f"  Cycles per A angle: { {k: v for k, v in sorted(a_dist.items())} }")

    forces = [c['peak_force_kn'] for c in cycles]
    strokes = [c['max_stroke_mm'] for c in cycles]
    print(f"  Force  (kN): mean={np.mean(forces):.1f}  min={np.min(forces):.1f}  max={np.max(forces):.1f}")
    print(f"  Stroke (mm): mean={np.mean(strokes):.1f}  min={np.min(strokes):.1f}  max={np.max(strokes):.1f}")

    if fit_results:
        a_fits = [r['a_px'] for r in fit_results.values()]
        b_fits = [r['b_px'] for r in fit_results.values()]
        residuals = [r['residual_px'] for r in fit_results.values()]
        print(f"\n  Ellipse fit results ({len(fit_results)} Y slices):")
        print(f"    semi-axis a: mean={np.mean(a_fits):.1f}px  range={np.min(a_fits):.1f}-{np.max(a_fits):.1f}px")
        print(f"    semi-axis b: mean={np.mean(b_fits):.1f}px  range={np.min(b_fits):.1f}-{np.max(b_fits):.1f}px")
        print(f"    RMS residual: mean={np.mean(residuals):.2f}px  max={np.max(residuals):.2f}px")
        print(f"    aspect ratio a/b: mean={np.mean(np.array(a_fits)/np.array(b_fits)):.3f}")
        print()
        print("  NOTE: pixel→mm scale not yet calibrated.")
        print("  To calibrate: match billet length (Y span) against known billet size,")
        print("  or use line-scanner reference at time=0.")

    print("\n  Output files in reconstruction_out/:")
    for p in sorted(OUT_DIR.glob('*.png')):
        print(f"    {p.name}")
    npz = OUT_DIR / 'point_cloud.npz'
    if npz.exists():
        print(f"    {npz.name}")


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cycles = build_cycle_table(MCAP_PATH)
    silhouettes = extract_silhouettes_for_cycles(MCAP_PATH, cycles)

    plot_sample_silhouettes(cycles, silhouettes, OUT_DIR, n=12)

    fit_results, groups = fit_cross_sections(cycles, silhouettes)

    if fit_results:
        plot_cross_section_summary(fit_results, groups, OUT_DIR)
        plot_width_vs_a(groups, OUT_DIR)
        cloud = build_point_cloud(fit_results, n_pts_per_slice=64)
        plot_point_cloud(cloud, OUT_DIR)
        if cloud is not None:
            np.savez(OUT_DIR / 'point_cloud.npz',
                     points=cloud,
                     y_keys=np.array(sorted(fit_results.keys())),
                     a_px=np.array([fit_results[y]['a_px'] for y in sorted(fit_results)]),
                     b_px=np.array([fit_results[y]['b_px'] for y in sorted(fit_results)]),
                     residuals=np.array([fit_results[y]['residual_px'] for y in sorted(fit_results)]))
            print(f"Saved point_cloud.npz ({len(cloud)} points)")
    else:
        print("WARNING: no ellipse fits succeeded — check silhouette extraction")
        # Still plot width data to diagnose
        plot_width_vs_a(groups, OUT_DIR)

    print_summary(cycles, fit_results)
