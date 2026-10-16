"""
Live webcam check and recorder.

Shows the hand skeleton, the pinch ratio (thumb-index distance / palm length) with
the resulting gripper state, and the four board compartments projected onto the image.
Overlays are drawn on a display copy only, not on the raw data used later. Detection always runs on the raw frame,
and recordings are saved without overlays.

Keys:  r = start/stop recording     q = quit
Recordings: data/raw/<name>_NN.mp4, numbered automatically.

Calling from the project root:
  python scripts/live_hand_check.py --name morning
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from board import BoardTracker, approx_intrinsics  # noqa: E402

MODEL = "models/hand_landmarker.task"
CLOSE_TH, OPEN_TH = 0.72, 0.85        # tuned on my hand: closed below CLOSE_TH, open above OPEN_TH
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
               (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
               (17, 18), (18, 19), (19, 20)]


def next_path(name):
    out = Path("data/raw"); out.mkdir(parents=True, exist_ok=True)
    i = 1
    while (out / f"{name}_{i:02d}.mp4").exists():
        i += 1
    return out / f"{name}_{i:02d}.mp4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="demo", help="compartment you are about to demonstrate")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--hfov", type=float, default=70.0, help="webcam horizontal field of view (deg)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read from webcam (another app may be using it).")
    h, w = frame.shape[:2]
    tracker = BoardTracker(approx_intrinsics(w, h, args.hfov))

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1)

    closed, recording, buffer, rec_start = False, False, [], 0.0
    t0 = prev = time.time()
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if recording:
                buffer.append(frame.copy())                     # raw frame, no overlays

            n_corners = tracker.update(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                              int((time.time() - t0) * 1000))

            view = tracker.draw_zones(frame)
            if res.hand_landmarks:
                pts = [(int(p.x * w), int(p.y * h)) for p in res.hand_landmarks[0]]
                for a, b in CONNECTIONS:
                    cv2.line(view, pts[a], pts[b], (0, 255, 0), 2)
                wl = res.hand_world_landmarks[0]
                d = lambda a, b: np.linalg.norm([wl[a].x - wl[b].x, wl[a].y - wl[b].y, wl[a].z - wl[b].z])
                ratio = d(4, 8) / d(0, 9)
                if not closed and ratio < CLOSE_TH:
                    closed = True
                elif closed and ratio > OPEN_TH:
                    closed = False
                cv2.putText(view, f"pinch {ratio:.2f} -> {'CLOSED' if closed else 'OPEN'}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            now = time.time()
            cv2.putText(view, f"{1 / max(now - prev, 1e-6):.0f} fps | board corners {n_corners}/24",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if n_corners >= 10 else (0, 0, 255), 2)
            prev = now
            if recording:
                cv2.circle(view, (w - 30, 30), 10, (0, 0, 255), -1)
                cv2.putText(view, f"REC {now - rec_start:4.1f}s", (w - 150, 37),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("hand check - r: record, q: quit", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                if not recording:
                    recording, buffer, rec_start = True, [], time.time()
                else:
                    recording = False
                    fps = len(buffer) / max(time.time() - rec_start, 1e-6)   # real frame rate
                    path = next_path(args.name)
                    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    for f in buffer:
                        vw.write(f)
                    vw.release()
                    print(f"saved {path}  ({len(buffer)} frames, {fps:.1f} fps)")
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()