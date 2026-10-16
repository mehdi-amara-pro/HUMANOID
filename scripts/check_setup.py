import cv2, numpy as np, mediapipe as mp, mujoco, scipy, rerun as rr

print("OpenCV", cv2.__version__, "| MediaPipe", mp.__version__, "| MuJoCo", mujoco.__version__)

# ArUco: generate a marker, then detect it
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker = cv2.aruco.generateImageMarker(d, 0, 400)
img = cv2.copyMakeBorder(marker, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
corners, ids, _ = cv2.aruco.ArucoDetector(d).detectMarkers(img)
print("ArUco detected IDs:", None if ids is None else ids.ravel())

# MuJoCo: build and step a tiny scene
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><freejoint/><geom size='.1'/></body></worldbody></mujoco>")
dat = mujoco.MjData(m); mujoco.mj_step(m, dat)
print("MuJoCo step OK")

# Rerun: open the viewer with a few 3D points
rr.init("setup_check", spawn=True)
rr.log("points", rr.Points3D(np.random.rand(20, 3)))
print("Rerun viewer should have opened")