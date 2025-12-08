import cv2
import numpy as np

def detect_clocks(frame):
    if frame is None or frame.size == 0:
        return None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1)

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
    
    # Need at least 16 clocks (12 equation + 4 answers)
    if len(circles) < 16:
        print(f"[Warning] Only found {len(circles)} circles, need 16")
        return None

    # Sort by y coordinate (top grid vs bottom answers)
    circles = sorted(circles, key=lambda c: c[1])

    top = circles[:12]    # equation clocks
    bottom = circles[12:16]  # answer clocks (only take first 4)

    eq_rois = []
    h, w = frame.shape[:2]
    
    for (x, y, r) in top:
        # Convert to int to prevent overflow warnings
        x, y, r = int(x), int(y), int(r)
        
        # Ensure ROI is within frame bounds
        y1 = max(0, y - r)
        y2 = min(h, y + r)
        x1 = max(0, x - r)
        x2 = min(w, x + r)
        
        # Only add if valid region
        if y2 > y1 and x2 > x1:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                eq_rois.append(roi)
            else:
                eq_rois.append(None)
        else:
            eq_rois.append(None)

    ans_rois = []
    for (x, y, r) in bottom:
        # Convert to int to prevent overflow warnings
        x, y, r = int(x), int(y), int(r)
        
        # Ensure ROI is within frame bounds
        y1 = max(0, y - r)
        y2 = min(h, y + r)
        x1 = max(0, x - r)
        x2 = min(w, x + r)
        
        # Only add if valid region
        if y2 > y1 and x2 > x1:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                ans_rois.append(roi)
            else:
                ans_rois.append(None)
        else:
            ans_rois.append(None)

    return eq_rois, ans_rois