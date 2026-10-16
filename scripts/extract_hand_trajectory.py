"""
Demo videos  ->  hand trajectories on the board  (w/ perception accuracy)

For every clip:
  1. Board pose from the ChArUco board (static camera: last valid pose is reused
     while the hand covers the board).
  2. Hand landmarks (MediaPipe), hand position in the camera frame by PnP between
     the metric hand model and its 2D projection.
  3. Grasp point (midpoint of thumb and index tips) in the table frame
     (x = right, y = far, z = up, origin at the board centre, metres).
  4. Pinch ratio = thumb-index distance / palm length, smoothed; grasp and release
     detected with hysteresis (longest closed segment = the carry).
  5. Release point -> compartment the system *perceives*, compared with the
     compartment in the filename (what was *intended*).

Outputs for the next steps
  trajectories/<clip>.npz           trajectory, pinch ratio, events, intended + perceived zone
  trajectories/<clip>.png           position + pinch ratio over time, events marked
  results/perception.csv            one row per clip
  results/release_points.png        top-down view of every release point on the board
  data/debug/<clip>_debug.mp4       overlay video (zones, skeleton, gripper state)

Calling
  python scripts/extract_hand_trajectory.py "data/raw/*.mp4"
"""
import argparse
import csv
import glob
import sys
from pathlib import Path
 
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt
 
sys.path.insert(0, str(Path(__file__).parent))
from board import BoardTracker, ZONES, approx_intrinsics, zone_of  # noqa: E402
 
MODEL = "models/hand_landmarker.task"
CLOSE_TH, OPEN_TH = 0.72, 0.85        # same values as the live check
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9
ZONE_COLORS = {"Morning": "#F2C14E", "Noon": "#F78154", "Afternoon": "#4D9DE0", "Night": "#5B5F97"}
 
 
def detect_events(ratio):
    """Grasp/release indices from the pinch ratio, with hysteresis.
    Returns (i_grasp, i_release, closed_mask) or raises if no grasp is found."""
    closed, state = np.zeros(len(ratio), dtype=bool), False
    for i, r in enumerate(ratio):
        if not state and r < CLOSE_TH:
            state = True
        elif state and r > OPEN_TH:
            state = False
        closed[i] = state
    idx = np.flatnonzero(closed)
    if len(idx) == 0:
        raise ValueError("no grasp detected")
    segments = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    carry = max(segments, key=len)
    return carry[0], carry[-1], closed
 
 
def fill_and_smooth(t, x):
    """Interpolate NaN gaps, remove single-frame spikes, smooth jitter (works on 1D or 2D)."""
    x = x.copy()
    cols = x if x.ndim == 2 else x[:, None]
    for k in range(cols.shape[1]):
        ok = ~np.isnan(cols[:, k])
        cols[:, k] = np.interp(t, t[ok], cols[ok, k])
        cols[:, k] = uniform_filter1d(medfilt(cols[:, k], 5), 5)
    return cols if x.ndim == 2 else cols[:, 0]
 
 
def process(video_path, hfov, grasp_height, traj_dir, debug_dir):
    stem = Path(video_path).stem
    intended = stem.split("_")[0].capitalize()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K = approx_intrinsics(w, h, hfov)
    tracker = BoardTracker(K)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1)
 
    frames, T, P, P_ray, Rt, board_seen = [], [], [], [], [], []
    K_inv = np.linalg.inv(K)
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as lm:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            board_seen.append(tracker.update(frame) > 0)
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int(t * 1000))
            pos, pos_ray, ratio, img_pts = np.full(3, np.nan), np.full(3, np.nan), np.nan, None
            if res.hand_landmarks and tracker.pose is not None:
                img_pts = np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]])
                # Method B: camera ray through the fingertips' pixel, intersected with the plane
                # z = grasp_height above the table (the board gives that plane exactly).
                uv = 0.5 * (img_pts[THUMB_TIP] + img_pts[INDEX_TIP])
                origin = tracker.to_table(np.zeros(3))
                direction = tracker.to_table(K_inv @ np.array([uv[0], uv[1], 1.0])) - origin
                if direction[2] < -1e-6:
                    pos_ray = origin + (grasp_height - origin[2]) / direction[2] * direction
                hand3d = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]])
                ok_h, rv, tv = cv2.solvePnP(hand3d, img_pts, K, np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
                if ok_h:
                    cam = (cv2.Rodrigues(rv)[0] @ hand3d.T).T + tv.reshape(3)
                    pos = tracker.to_table(0.5 * (cam[THUMB_TIP] + cam[INDEX_TIP]))
                ratio = (np.linalg.norm(hand3d[THUMB_TIP] - hand3d[INDEX_TIP])
                         / np.linalg.norm(hand3d[WRIST] - hand3d[MIDDLE_MCP]))
            frames.append((frame, img_pts))
            T.append(t); P.append(pos); P_ray.append(pos_ray); Rt.append(ratio)
            i += 1
    cap.release()
 
    T, P, P_ray, Rt = np.array(T), np.array(P), np.array(P_ray), np.array(Rt)
    tracked = ~np.isnan(P).any(axis=1)
    if tracked.sum() < 10 or (~np.isnan(P_ray).any(axis=1)).sum() < 10:
        print(f"{stem}: hand tracked in only {tracked.sum()} frames, skipped")
        return None
    Ps, Ps_ray, Rs = fill_and_smooth(T, P), fill_and_smooth(T, P_ray), fill_and_smooth(T, Rt)
 
    try:
        i_g, i_r, closed = detect_events(Rs)
        release_pnp, release = Ps[i_r], Ps_ray[i_r]          # A: hand-size depth, B: ray-table
        perceived_pnp = zone_of(release_pnp[0], release_pnp[1])
        perceived = zone_of(release[0], release[1])
    except ValueError:
        i_g = i_r = -1
        closed = np.zeros(len(T), dtype=bool)
        release = release_pnp = np.full(3, np.nan)
        perceived = perceived_pnp = "none"
    hr, hf = tracker.half_extents()
 
    # Debug video: zones, skeleton, gripper state, event flashes
    vw = cv2.VideoWriter(str(debug_dir / f"{stem}_debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for k, (frame, img_pts) in enumerate(frames):
        view = tracker.draw_zones(frame)
        if img_pts is not None:
            for x, y in img_pts.astype(int):
                cv2.circle(view, (x, y), 3, (0, 0, 255), -1)
        state = "CLOSED" if closed[k] else "OPEN"
        cv2.putText(view, f"{stem} | pinch {Rs[k]:.2f} -> {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if i_g >= 0 and abs(k - i_g) < 5:
            cv2.putText(view, "GRASP", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if i_r >= 0 and abs(k - i_r) < 5:
            cv2.putText(view, f"RELEASE -> {perceived}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        vw.write(view)
    vw.release()
 
    np.savez(traj_dir / f"{stem}.npz", t=T, pos=Ps, pos_ray=Ps_ray, ratio=Rs, tracked=tracked,
             i_grasp=i_g, i_release=i_r, release=release, intended=intended, perceived=perceived,
             perceived_pnp=perceived_pnp, half_extents=[hr, hf], fps=fps, hfov=hfov,
             grasp_height=grasp_height)
 
    # Per-clip plot
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for k, name in enumerate(["right", "far", "up"]):
        ax[0].plot(T, Ps[:, k] * 100, label=name)
    ax[0].set_ylabel("grasp point [cm]"); ax[0].legend(loc="upper right"); ax[0].grid(alpha=0.3)
    ax[1].plot(T, Rs, color="k", label="pinch ratio")
    ax[1].axhline(CLOSE_TH, ls="--", color="tab:green", lw=1, label="close threshold")
    ax[1].axhline(OPEN_TH, ls="--", color="tab:red", lw=1, label="open threshold")
    for a in ax:
        if i_g >= 0:
            a.axvline(T[i_g], color="tab:green", lw=1.5)
            a.axvline(T[i_r], color="tab:red", lw=1.5)
    ax[1].set_ylabel("pinch ratio"); ax[1].set_xlabel("time [s]"); ax[1].legend(loc="upper right", fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"{stem}: intended {intended}, perceived {perceived}, hand tracked {tracked.mean():.0%}")
    fig.tight_layout(); fig.savefig(traj_dir / f"{stem}.png", dpi=120); plt.close(fig)
 
    print(f"{stem:14s} intended {intended:9s} | hand-size depth: {perceived_pnp:9s} "
          f"{'OK ' if perceived_pnp == intended else 'MISS'} | ray-table: {perceived:9s} "
          f"{'OK ' if perceived == intended else 'MISS'} | tracked {tracked.mean():.0%}")
    return dict(clip=stem, intended=intended, perceived=perceived, correct=int(perceived == intended),
                perceived_pnp=perceived_pnp, correct_pnp=int(perceived_pnp == intended),
                release_right_cm=round(100 * release[0], 1), release_far_cm=round(100 * release[1], 1),
                pnp_right_cm=round(100 * release_pnp[0], 1), pnp_far_cm=round(100 * release_pnp[1], 1),
                pnp_up_cm=round(100 * release_pnp[2], 1), duration_s=round(T[-1], 2),
                grasp_s=round(T[i_g], 2) if i_g >= 0 else None,
                release_s=round(T[i_r], 2) if i_r >= 0 else None,
                hand_tracked=round(float(tracked.mean()), 2), half_right_cm=round(100 * hr, 1),
                half_far_cm=round(100 * hf, 1))
 
 
def release_figure(rows, out_path):
    """Top-down view of the board with every release point, coloured by intended compartment."""
    hr, hf = rows[0]["half_right_cm"], rows[0]["half_far_cm"]
    fig, ax = plt.subplots(figsize=(5, 6.5))
    quads = {"Morning": (-hr, 0), "Noon": (0, 0), "Afternoon": (-hr, -hf), "Night": (0, -hf)}
    for name, (x0, y0) in quads.items():
        ax.add_patch(Rectangle((x0, y0), hr, hf, color=ZONE_COLORS[name], alpha=0.25, lw=0))
        ax.text(x0 + hr / 2, y0 + hf - 1.5, name, ha="center", va="top", fontsize=10)
    ax.add_patch(Rectangle((-hr, -hf), 2 * hr, 2 * hf, fill=False, lw=1.2))
    ax.axhline(0, color="k", ls="--", lw=0.8); ax.axvline(0, color="k", ls="--", lw=0.8)
    for r in rows:
        if r["perceived"] == "none":
            continue
        c = ZONE_COLORS[r["intended"]]
        # faint: method A (hand-size depth); arrow to method B (ray-table intersection)
        ax.scatter(r["pnp_right_cm"], r["pnp_far_cm"], s=30, color=c, alpha=0.35, edgecolor="none", zorder=2)
        ax.annotate("", xy=(r["release_right_cm"], r["release_far_cm"]),
                    xytext=(r["pnp_right_cm"], r["pnp_far_cm"]),
                    arrowprops=dict(arrowstyle="->", color="0.5", lw=0.7), zorder=2)
        ax.scatter(r["release_right_cm"], r["release_far_cm"], s=60, color=c,
                   edgecolor="k", marker="o" if r["correct"] else "X", zorder=3)
    n_ok, n_pnp = sum(r["correct"] for r in rows), sum(r["correct_pnp"] for r in rows)
    ax.set_title(f"Release points: hand-size depth {n_pnp}/{len(rows)} -> ray-table {n_ok}/{len(rows)} correct\n"
                 "faint = hand-size depth, solid = ray-table, colour = intended, X = wrong", fontsize=9)
    ax.set_xlabel("right [cm]  (camera's view)"); ax.set_ylabel("far [cm]")
    off = sum(1 for r in rows if abs(r["release_right_cm"]) > hr + 3 or abs(r["release_far_cm"]) > hf + 3)
    if off:
        ax.text(0, -hf - 2, f"{off} release point(s) outside the board, not shown",
                ha="center", va="top", fontsize=8, color="0.3")
    ax.set_aspect("equal"); ax.set_xlim(-hr - 3, hr + 3); ax.set_ylim(-hf - 4, hf + 3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="video file(s) or glob pattern(s)")
    ap.add_argument("--hfov", type=float, default=70.0, help="webcam horizontal field of view (deg)")
    ap.add_argument("--grasp_height", type=float, default=0.03,
                    help="height of the fingertips above the table at release, i.e. box height (m)")
    args = ap.parse_args()
 
    traj_dir, debug_dir, res_dir = Path("trajectories"), Path("data/debug"), Path("results")
    for d in (traj_dir, debug_dir, res_dir):
        d.mkdir(parents=True, exist_ok=True)
    paths = sorted({p for pattern in args.videos for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit("No videos found.")
 
    rows = [r for r in (process(p, args.hfov, args.grasp_height, traj_dir, debug_dir) for p in paths) if r]
    with open(res_dir / "perception.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
    release_figure(rows, res_dir / "release_points.png")
 
    n_ok, n_pnp = sum(r["correct"] for r in rows), sum(r["correct_pnp"] for r in rows)
    print(f"\nPerception accuracy (clips in the intended compartment)")
    print(f"  {'':9s} hand-size depth   ray-table")
    print(f"  {'All':9s} {n_pnp:>6}/{len(rows):<10} {n_ok:>3}/{len(rows)}")
    for z in ZONES:
        zr = [r for r in rows if r["intended"] == z]
        if zr:
            print(f"  {z:9s} {sum(r['correct_pnp'] for r in zr):>6}/{len(zr):<10} "
                  f"{sum(r['correct'] for r in zr):>3}/{len(zr)}")
    print("Saved results/perception.csv and results/release_points.png")
 
 
if __name__ == "__main__":
    main()