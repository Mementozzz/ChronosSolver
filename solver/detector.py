import cv2
import numpy as np

def detect_clocks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 1)

    edges = cv2.Canny(blur, 50, 150)

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=70,
        param1=100,
        param2=25,
        minRadius=35,
        maxRadius=120
    )

    if circles is None:
        return None

    circles = np.uint16(np.around(circles))[0]

    # Sort by y coordinate (top grid vs bottom answers)
    circles = sorted(circles, key=lambda c: c[1])

    top = circles[:12]    # equation clocks
    bottom = circles[12:] # answer clocks

    eq_rois = []
    for (x,y,r) in top:
        roi = frame[y-r:y+r, x-r:x+r]
        eq_rois.append(roi)

    ans_rois = []
    for (x,y,r) in bottom:
        roi = frame[y-r:y+r, x-r:x+r]
        ans_rois.append(roi)

    return eq_rois, ans_rois