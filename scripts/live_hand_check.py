import time
import cv2
import numpy as np
import mediapipe as mp

MODEL = "models/hand_landmarker.task"
CLOSE_TH, OPEN_TH = 0.35, 0.50   # tune these by watching the ratio on screen
closed = False
CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
               (9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
)

cap = cv2.VideoCapture(0)
t0 = prev = time.time()

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read from webcam"); break
        frame = cv2.flip(frame, 1)                      # mirror view
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - t0) * 1000)          # must increase every frame
        result = landmarker.detect_for_video(image, ts_ms)

        h, w = frame.shape[:2]
        if result.hand_landmarks:
            pts = [(int(p.x * w), int(p.y * h)) for p in result.hand_landmarks[0]]
            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for p in pts:
                cv2.circle(frame, p, 4, (0, 0, 255), -1)

            # Pinch as a ratio of palm length: scale-free, independent of the other fingers
            wl = result.hand_world_landmarks[0]
            def d(a, b):
                return np.linalg.norm([wl[a].x - wl[b].x, wl[a].y - wl[b].y, wl[a].z - wl[b].z])
            ratio = d(4, 8) / d(0, 9)          # thumb-index / wrist-middle knuckle

            # Hysteresis: close below CLOSE_TH, re-open only above OPEN_TH (no flicker)
            if not closed and ratio < CLOSE_TH:
                closed = True
            elif closed and ratio > OPEN_TH:
                closed = False
            cv2.putText(frame, f"pinch ratio {ratio:.2f} -> gripper {'CLOSED' if closed else 'OPEN'}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "no hand", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        now = time.time()
        cv2.putText(frame, f"{1 / max(now - prev, 1e-6):.0f} fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        prev = now

        cv2.imshow("hand check - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()