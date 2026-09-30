import sys
import cv2
import numpy as np

path = sys.argv[1]
cap = cv2.VideoCapture(path)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
first = last = None
for i in range(n):
    ok, f = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if i == 0:
        first = g
    last = g
cap.release()
h, w = first.shape
win = cv2.createHanningWindow((w, h), cv2.CV_32F)
(dx, dy), resp = cv2.phaseCorrelate(first, last, win)
mag = (dx * dx + dy * dy) ** 0.5
print("frames", n, "| shift dx,dy:", round(dx, 1), round(dy, 1), "| mag:", round(mag, 1), "| resp:", round(float(resp), 3))
