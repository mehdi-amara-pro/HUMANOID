"""
Frames
  board object frame  OpenCV's convention: origin at the board's outer corner,
                      x along the 5 columns, y along the 7 rows, z into the paper.
  table frame         camera-aligned and independent of how the board was laid down:
                      origin at the board centre, x = right (as seen from the camera),
                      y = far (away from the camera), z = up.  Units: m

Compartments are the four equal quadrants of the board in the table frame:
Morning is far-left, Noon is far-right, Afternoon is near-left, Night is near-right
"""
import cv2
import numpy as np

SQUARE_M, MARKER_M = 0.030, 0.022          # measured on the printed board
COLS, ROWS = 5, 7
BOARD_W, BOARD_H = COLS * SQUARE_M, ROWS * SQUARE_M
ZONES = ["Morning", "Noon", "Afternoon", "Night"]
ZONE_COLORS_BGR = {"Morning": (78, 193, 242), "Noon": (84, 129, 247),
                   "Afternoon": (224, 157, 77), "Night": (151, 95, 91)}
F_FLIP = np.diag([1.0, -1.0, -1.0])        # board object frame -> right-handed, z-up frame


def make_board():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    return cv2.aruco.CharucoBoard((COLS, ROWS), SQUARE_M, MARKER_M, d)


def approx_intrinsics(w, h, hfov_deg):
    """Pinhole intrinsics from the horizontal field of view (no calibration)."""
    f = (w / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


class BoardTracker:
    """Detects the board in each frame and converts camera-frame points to the table frame."""

    def __init__(self, K, min_corners=6):
        self.board = make_board()
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.K, self.dist = K, np.zeros(5)
        self.min_corners = min_corners
        self.pose = None          # (R, t): board object frame -> camera frame
        self.A = None             # table-axes rotation, fixed at the first detection
        self.centre = np.array([BOARD_W / 2, -BOARD_H / 2, 0.0])   # board centre, flipped frame

    def update(self, frame_bgr):
        """Returns the number of ChArUco corners used (0 if the board was not found)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _, _ = self.detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < self.min_corners:
            return 0
        obj, img = self.board.matchImagePoints(ch_corners, ch_ids)
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.dist)
        if not ok:
            return 0
        R, t = cv2.Rodrigues(rvec)[0], tvec.reshape(3)
        self.pose = (R, t)
        if self.A is None:
            self._fix_axes(R, t)
        return len(ch_ids)

    def _fix_axes(self, R, t):
        """Choose 'right' and 'far' from where the camera is, so zone names match the camera's view."""
        cam = F_FLIP @ (-R.T @ t) - self.centre             # camera position, board-centred, z up
        if cam[2] < 0:
            print("warning: camera appears below the board plane, check the board detection")
        away = -cam[:2] / np.linalg.norm(cam[:2])           # away from the camera, in the board plane
        # Snap to the board edge direction closest to "away", so compartments follow the printed grid
        # even when the camera is off to one side of the board.
        candidates = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        far = candidates[np.argmax(candidates @ away)]
        right = np.array([far[1], -far[0]])                 # far x up
        self.A = np.array([[right[0], right[1], 0], [far[0], far[1], 0], [0, 0, 1]])

    def to_table(self, p_cam):
        """Camera-frame point (m) -> table frame (right, far, up) in metres."""
        R, t = self.pose
        p_obj = R.T @ (p_cam - t)
        return self.A @ (F_FLIP @ p_obj - self.centre)

    def table_to_image(self, pts_table):
        """Table-frame points -> pixel coordinates (for drawing the overlay)."""
        R, t = self.pose
        p_obj = (F_FLIP @ (self.A.T @ np.asarray(pts_table).T + self.centre[:, None])).T
        px, _ = cv2.projectPoints(p_obj, cv2.Rodrigues(R)[0], t, self.K, self.dist)
        return px.reshape(-1, 2)

    def half_extents(self):
        """Half-size of the board along the table's right and far axes (m)."""
        corners = np.array([[0, 0, 0], [BOARD_W, 0, 0], [BOARD_W, -BOARD_H, 0], [0, -BOARD_H, 0]])
        tab = (self.A @ (corners - self.centre).T).T
        return np.abs(tab[:, 0]).max(), np.abs(tab[:, 1]).max()

    def draw_zones(self, img, alpha=0.30):
        """Paint the four compartments on a display copy of the frame."""
        if self.pose is None or self.A is None:
            return img
        hr, hf = self.half_extents()
        quads = {"Morning": (-hr, 0, 0, hf), "Noon": (0, hr, 0, hf),
                 "Afternoon": (-hr, 0, -hf, 0), "Night": (0, hr, -hf, 0)}
        layer = img.copy()
        for name, (x0, x1, y0, y1) in quads.items():
            poly = self.table_to_image([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
            cv2.fillPoly(layer, [poly.astype(np.int32)], ZONE_COLORS_BGR[name])
        out = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)
        for name, (x0, x1, y0, y1) in quads.items():
            c = self.table_to_image([[(x0 + x1) / 2, (y0 + y1) / 2, 0]])[0].astype(int)
            cv2.putText(out, name, (c[0] - 40, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return out


def zone_of(right, far):
    """Compartment name for a point in the table frame."""
    if far >= 0:
        return "Morning" if right < 0 else "Noon"
    return "Afternoon" if right < 0 else "Night"