import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

A4 = (8.27, 11.69)          # inches
MM = 1 / 25.4               # mm -> inches

def page(pdf, img, w_mm, h_mm, caption):
    fig = plt.figure(figsize=A4)
    w, h = w_mm * MM / A4[0], h_mm * MM / A4[1]
    ax = fig.add_axes([(1 - w) / 2, (1 - h) / 2 + 0.04, w, h])
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.axis("off")
    # 100 mm reference bar: measure it after printing to confirm the scale
    x0, y = (1 - 100 * MM / A4[0]) / 2, 0.10
    fig.add_artist(plt.Line2D([x0, x0 + 100 * MM / A4[0]], [y, y], transform=fig.transFigure, lw=2, color="k"))
    fig.text(0.5, 0.11, "this bar must measure 100 mm", ha="center", fontsize=9)
    fig.text(0.5, 0.05, caption + "  |  PRINT AT 100% / ACTUAL SIZE", ha="center", fontsize=10)
    pdf.savefig(fig); plt.close(fig)

with PdfPages("aruco_print.pdf") as pdf:
    # Page 1: scene marker, DICT_4X4_50, ID 0, 150 mm black square
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    page(pdf, cv2.aruco.generateImageMarker(d, 0, 600), 150, 150,
         "ArUco DICT_4X4_50  ID 0  side 150 mm")

    # Page 2: ChArUco calibration board, 5x7 squares, 30 mm squares, 22 mm markers
    d2 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((5, 7), 0.030, 0.022, d2)
    page(pdf, board.generateImage((1500, 2100), marginSize=0), 150, 210,
         "ChArUco 5x7  square 30 mm  marker 22 mm  DICT_5X5_100")

print("Saved aruco_print.pdf")