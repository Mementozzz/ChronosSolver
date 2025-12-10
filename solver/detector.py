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
        
        # Clock faces should have 15-45% white pixels (border + ticks)
        if not (0.15 < white_ratio < 0.45):
            return False
        
        # Additional check: look for circular edges using Hough on the ROI itself
        edges = cv2.Canny(gray, 50, 150)
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=15,
            minRadius=int(roi.shape[0] * 0.35),
            maxRadius=int(roi.shape[0] * 0.55)
        )
        
        # Should detect a circle within the ROI (the clock border)
        if circles is None:
            return False
        
        return True
    except:
        return False

def detect_clocks(frame, force_difficulty=None):
    if frame is None or frame.size == 0:
        return None
    
    h, w = frame.shape[:2]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    edges = cv2.Canny(blur, 30, 100)

    # Adjust parameters based on forced difficulty
    if force_difficulty == 2:
        # 2-star: very large clocks
        print("[Info] Forced 2-star mode - looking for large clocks")
        if w >= 2560:
            min_radius = 100
            max_radius = 200
            min_dist = 150
        else:
            min_radius = 85
            max_radius = 170
            min_dist = 130
        
        # Very strict area for 2-star: clocks are in upper-center area
        game_area_x_min = int(w * 0.2)
        game_area_x_max = int(w * 0.8)
        game_area_y_min = int(h * 0.1)
        game_area_y_max = int(h * 0.5)  # Only upper half
        
    elif force_difficulty == 3:
        # 3-star: smaller clocks
        print("[Info] Forced 3-star mode - looking for small clocks")
        if w >= 2560:
            min_radius = 50
            max_radius = 90
            min_dist = 70
        else:
            min_radius = 40
            max_radius = 75
            min_dist = 60
        
        # Wider area for 3-star: clocks fill more of the screen
        game_area_x_min = int(w * 0.1)
        game_area_x_max = int(w * 0.9)
        game_area_y_min = int(h * 0.05)
        game_area_y_max = int(h * 0.65)
        
    else:
        # Auto-detect mode
        if w >= 2560:
            min_radius = 50
            max_radius = 180
            min_dist = 100
        elif w >= 1920:
            min_radius = 40
            max_radius = 140
            min_dist = 80
        else:
            min_radius = 35
            max_radius = 110
            min_dist = 70
        
        game_area_x_min = int(w * 0.15)
        game_area_x_max = int(w * 0.85)
        game_area_y_min = int(w * 0.1)
        game_area_y_max = int(h * 0.7)

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=80,
        param2=35,  # Increased from 30 to be more selective
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return None

    circles = np.uint16(np.around(circles))[0]
    
    print(f"[Debug] Found {len(circles)} initial circles")
    
    # If difficulty is forced, skip auto-detection
    if force_difficulty:
        circles_to_use = circles
    else:
        # Group circles by size to determine difficulty
        radii = [r for (x, y, r) in circles]
        avg_radius = np.mean(radii)
        
        print(f"[Debug] Average radius: {avg_radius:.1f}px")
        
        # For 2-star, clocks are MUCH larger (typically 100+ pixels at 1080p)
        # For 3-star, clocks are smaller (typically 60-80 pixels at 1080p)
        # Filter by size based on expected difficulty
        
        if w >= 2560:  # 1440p+
            large_clock_threshold = 120
            small_clock_threshold = 80
        else:  # 1080p
            large_clock_threshold = 90
            small_clock_threshold = 60
        
        # Separate large and small circles
        large_circles = [(x, y, r) for (x, y, r) in circles if r >= large_clock_threshold]
        medium_circles = [(x, y, r) for (x, y, r) in circles if small_clock_threshold <= r < large_clock_threshold]
        
        print(f"[Debug] Large circles (r>={large_clock_threshold}): {len(large_circles)}")
        print(f"[Debug] Medium circles ({small_clock_threshold}<=r<{large_clock_threshold}): {len(medium_circles)}")
        
        # If we have large circles, this is likely 2-star or 1-star
        # If we only have medium circles, this is likely 3-star
        if len(large_circles) >= 5:
            # 2-star or 1-star - use only large circles
            circles_to_use = large_circles
            print(f"[Debug] Using large circles only (2-star or 1-star suspected)")
        elif len(medium_circles) >= 10:
            # 3-star - use medium circles
            circles_to_use = medium_circles
            print(f"[Debug] Using medium circles only (3-star suspected)")
        else:
            # Mixed or unclear - use both
            circles_to_use = large_circles + medium_circles
            print(f"[Debug] Using mixed circles")
    
    circles = circles_to_use
    
    # Define the main game area (clocks are typically in the center/upper-center of screen)
    # Exclude the very edges where UI elements are
    # (game_area already defined above based on difficulty)
    
    # Filter circles to only include clock-like ones in the game area
    valid_circles = []
    for (x, y, r) in circles:
        x, y, r = int(x), int(y), int(r)
        
        # Skip circles outside the main game area
        if not (game_area_x_min < x < game_area_x_max and game_area_y_min < y < game_area_y_max):
            continue
        
        # Extract ROI for validation
        y1 = max(0, y - r)
        y2 = min(h, y + r)
        x1 = max(0, x - r)
        x2 = min(w, x + r)
        
        if y2 > y1 and x2 > x1:
            roi = frame[y1:y2, x1:x2]
            if is_clock_like(roi):
                valid_circles.append((x, y, r))
    
    print(f"[Debug] Filtered to {len(valid_circles)} valid clock-like circles")
    
    if len(valid_circles) < 5:
        print(f"[Warning] Only found {len(valid_circles)} valid clock circles (need at least 5)")
        return None
    
    circles = valid_circles
    
    # Sort by Y position to find the vertical gap between equation and answer clocks
    circles_sorted = sorted(circles, key=lambda c: c[1])
    
    # Find the largest Y-gap (indicates separation between equation and answer rows)
    y_gaps = []
    for i in range(len(circles_sorted) - 1):
        gap = circles_sorted[i + 1][1] - circles_sorted[i][1]
        y_gaps.append((gap, i + 1))  # (gap_size, split_index)
    
    # Find the largest gap
    if y_gaps:
        y_gaps.sort(reverse=True)
        largest_gap_idx = y_gaps[0][1]
        
        # Split at the largest gap
        equation_circles = circles_sorted[:largest_gap_idx]
        answer_circles = circles_sorted[largest_gap_idx:]
        
        # Ensure we have exactly 4 answer circles (take bottom 4)
        if len(answer_circles) > 4:
            answer_circles = answer_circles[-4:]
        
        num_equation = len(equation_circles)
    else:
        # Fallback: assume bottom 4 are answers
        answer_circles = circles_sorted[-4:]
        equation_circles = circles_sorted[:-4]
        num_equation = len(equation_circles)
    
    print(f"[Debug] Split into {num_equation} equation circles and {len(answer_circles)} answer circles")
    
    # Determine difficulty
    if force_difficulty:
        difficulty = force_difficulty
        if difficulty == 2:
            num_to_use = 2
        else:  # difficulty == 3
            num_to_use = 15
    elif num_equation >= 10:
        difficulty = 3
        num_to_use = 15
    elif num_equation == 2:
        difficulty = 2
        num_to_use = 2
    elif num_equation == 1:
        difficulty = 1
        num_to_use = 1
    else:
        print(f"[Warning] Unexpected number of equation clocks: {num_equation}, defaulting to 2-star")
        difficulty = 2
        num_to_use = min(num_equation, 2)

    print(f"[Info] Detected {difficulty}-star difficulty ({num_to_use} equation clock{'s' if num_to_use > 1 else ''})")
    
    # Sort equation clocks by position (left-to-right, top-to-bottom)
    equation_circles = sorted(equation_circles, key=lambda c: (c[1] // 100, c[0]))
    
    # Sort answer clocks left-to-right
    answer_circles = sorted(answer_circles, key=lambda c: c[0])
    
    # Limit to number needed + 1 for result clock
    equation_circles = equation_circles[:num_to_use + 1]
    answer_circles = answer_circles[:4]
    
    # Extract ROIs for equation clocks
    eq_rois = []
    for (x, y, r) in equation_circles:
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
    for (x, y, r) in answer_circles:
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