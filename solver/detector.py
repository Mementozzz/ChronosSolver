import cv2
import numpy as np

def detect_clocks(frame):
    if frame is None or frame.size == 0:
        return None
    
    h, w = frame.shape[:2]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    edges = cv2.Canny(blur, 30, 100)

    # Adjust parameters based on resolution
    # For 1080p, clocks are larger
    if w >= 1920:  # 1080p or higher
        min_radius = 40
        max_radius = 100
        min_dist = 80
    else:  # Lower resolutions
        min_radius = 35
        max_radius = 120
        min_dist = 70

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=80,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return None

    circles = np.uint16(np.around(circles))[0]
    
    # Auto-detect difficulty based on number of circles found
    total_circles = len(circles)
    
    if total_circles >= 19:
        # 3-star: 15 equation clocks + 1 result + 4 answers = 20 total
        difficulty = 3
        num_equation = 15
        num_answers = 4
    elif total_circles >= 6:
        # 2-star: 2 equation clocks + 1 result + 4 answers = 7 total
        difficulty = 2
        num_equation = 2
        num_answers = 4
    elif total_circles >= 5:
        # 1-star: 1 clock + 4 answers = 5 total (no addition, just matching)
        difficulty = 1
        num_equation = 1
        num_answers = 4
    else:
        print(f"[Warning] Only found {total_circles} circles, need at least 5")
        return None

    print(f"[Info] Detected {difficulty}-star difficulty ({num_equation} equation clock{'s' if num_equation > 1 else ''})")

    # Sort by y coordinate (top grid vs bottom answers)
    circles = sorted(circles, key=lambda c: c[1])

    # Split based on detected difficulty
    top = circles[:num_equation + 1]  # equation clocks + result clock
    bottom = circles[num_equation + 1:num_equation + 1 + num_answers]  # answer clocks

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

    return eq_rois, ans_rois, difficulty
