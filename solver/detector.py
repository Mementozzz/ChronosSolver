import cv2
import numpy as np

def is_clock_like(roi):
    """Check if a ROI looks like a clock face with tick marks"""
    if roi is None or roi.size == 0 or roi.shape[0] < 30 or roi.shape[1] < 30:
        return False
    
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Look for white/bright circular edges (clock border and tick marks)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Count white pixels (should have clock border + tick marks)
        white_pixels = np.sum(thresh == 255)
        total_pixels = thresh.size
        white_ratio = white_pixels / total_pixels
        
        # Clock faces should have 15-40% white pixels (border + ticks)
        # Too few = not a clock, too many = solid white circle
        return 0.15 < white_ratio < 0.40
    except:
        return False

def detect_clocks(frame):
    if frame is None or frame.size == 0:
        return None
    
    h, w = frame.shape[:2]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    edges = cv2.Canny(blur, 30, 100)

    # Adjust parameters based on resolution
    if w >= 2560:  # 1440p or higher
        min_radius = 50
        max_radius = 150
        min_dist = 100
    elif w >= 1920:  # 1080p
        min_radius = 40
        max_radius = 120
        min_dist = 80
    else:  # Lower resolutions
        min_radius = 35
        max_radius = 100
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
    
    # Filter circles to only include clock-like ones
    valid_circles = []
    for (x, y, r) in circles:
        x, y, r = int(x), int(y), int(r)
        
        # Extract ROI for validation
        y1 = max(0, y - r)
        y2 = min(h, y + r)
        x1 = max(0, x - r)
        x2 = min(w, x + r)
        
        if y2 > y1 and x2 > x1:
            roi = frame[y1:y2, x1:x2]
            if is_clock_like(roi):
                valid_circles.append((x, y, r))
    
    if len(valid_circles) < 5:
        print(f"[Warning] Only found {len(valid_circles)} valid clock circles (need at least 5)")
        return None
    
    circles = valid_circles
    
    # Auto-detect difficulty based on number of circles found
    total_circles = len(circles)
    
    # Sort by Y position first to separate top and bottom rows
    circles_sorted = sorted(circles, key=lambda c: c[1])
    
    # Find the Y-gap between equation clocks and answer clocks
    # Answer clocks are always at the bottom
    y_positions = [c[1] for c in circles_sorted]
    
    # Take bottom 4 as answers (they should be clearly separated)
    answer_circles = circles_sorted[-4:]
    equation_circles = circles_sorted[:-4]
    
    num_equation = len(equation_circles)
    
    if num_equation >= 15:
        difficulty = 3
        num_to_use = 15
    elif num_equation >= 2:
        difficulty = 2
        num_to_use = 2
    elif num_equation >= 1:
        difficulty = 1
        num_to_use = 1
    else:
        print(f"[Warning] Not enough equation clocks: {num_equation}")
        return None

    print(f"[Info] Detected {difficulty}-star difficulty ({num_to_use} equation clock{'s' if num_to_use > 1 else ''})")
    print(f"[Debug] Found {len(equation_circles)} equation circles, {len(answer_circles)} answer circles")
    
    # Sort equation clocks by position (left-to-right, top-to-bottom)
    equation_circles = sorted(equation_circles, key=lambda c: (c[1] // 50, c[0]))
    
    # Sort answer clocks left-to-right
    answer_circles = sorted(answer_circles, key=lambda c: c[0])
    
    # Extract ROIs for equation clocks
    eq_rois = []
    for (x, y, r) in equation_circles[:num_to_use + 1]:  # +1 for result clock
        x, y, r = int(x), int(y), int(r)
        
        y1 = max(0, y - r)
        y2 = min(h, y + r)
        x1 = max(0, x - r)
        x2 = min(w, x + r)
        
        if y2 > y1 and x2 > x1:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                eq_rois.append(roi)
            else:
                eq_rois.append(None)
        else:
            eq_rois.append(None)

    # Extract ROIs for answer clocks
    ans_rois = []
    for (x, y, r) in answer_circles[:4]:
        x, y, r = int(x), int(y), int(r)
        
        y1 = max(0, y - r)
        y2 = min(h, y + r)
        x1 = max(0, x - r)
        x2 = min(w, x + r)
        
        if y2 > y1 and x2 > x1:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                ans_rois.append(roi)
            else:
                ans_rois.append(None)
        else:
            ans_rois.append(None)

    return eq_rois, ans_rois, difficulty