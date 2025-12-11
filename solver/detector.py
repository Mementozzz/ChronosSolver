import cv2
import numpy as np

def is_clock_like(roi, difficulty=None):
    """Check if a ROI looks like a clock face with tick marks"""
    if roi is None or roi.size == 0 or roi.shape[0] < 30 or roi.shape[1] < 30:
        return False
    
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Check for white pixels (clock border + tick marks)
        _, thresh_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        white_pixels = np.sum(thresh_white == 255)
        total_pixels = thresh_white.size
        white_ratio = white_pixels / total_pixels
        
        # Method 2: Check for edge density (works better on bright backgrounds)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges == 255)
        edge_ratio = edge_pixels / total_pixels
        
        # For 3-star clocks
        if difficulty == 3:
            # Accept if either white pixels OR edges indicate a clock
            has_white_features = 0.10 < white_ratio < 0.50
            has_edge_features = 0.05 < edge_ratio < 0.25
            
            if not (has_white_features or has_edge_features):
                return False
                
            # Additional check: should have some circular structure
            circles = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=100,
                param1=30,  # Lower threshold
                param2=10,  # Lower threshold for small clocks
                minRadius=int(roi.shape[0] * 0.3),
                maxRadius=int(roi.shape[0] * 0.6)
            )
            
            # For 3-star, we're more lenient - don't require circle detection
            return True
            
        else:
            # 2-star clocks should have clear borders
            if not (0.15 < white_ratio < 0.45):
                return False
            
            # Additional check: look for circular edges
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

    # 1. Define General Hough Search Parameters (Broad range to catch all clocks)
    if w >= 2560:
        hough_min_radius = 50
        hough_max_radius = 180
        hough_min_dist = 100
    elif w >= 1920:
        hough_min_radius = 40
        hough_max_radius = 140
        hough_min_dist = 80
    else:
        hough_min_radius = 35
        hough_max_radius = 110
        hough_min_dist = 70
    
    circles_raw = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=hough_min_dist,
        param1=80,
        param2=25,
        minRadius=hough_min_radius,
        maxRadius=hough_max_radius
    )

    if circles_raw is None:
        return None

    circles = np.uint16(np.around(circles_raw))[0]
    
    print(f"[Debug] Found {len(circles)} initial circles")
    
    # 2. Determine Difficulty and Filter Circles
    
    # Define size thresholds based on resolution
    if w >= 2560:
        large_clock_threshold = 120
        small_clock_threshold = 80
    else: # 1920x1080
        large_clock_threshold = 90
        small_clock_threshold = 60
    
    # TWEAK 2: Adjust logic for forced 3-star to include both small equation clocks and large answer clocks
    if force_difficulty == 3:
        difficulty = 3
        
        # When forcing 3-star, we must accept all clocks: small (equation) and large (answer).
        # We assume any clock larger than a tiny artifact is a candidate.
        circles_to_use = [(x, y, r) for (x, y, r) in circles if r >= 40]
        
        # Set a very broad search area for 3-star to catch all 15 equation clocks and 4 answers
        game_area_x_min = int(w * 0.05)
        game_area_x_max = int(w * 0.95)
        game_area_y_min = int(h * 0.05)
        game_area_y_max = int(h * 0.95)
        
    elif force_difficulty == 2:
        difficulty = 2
        # Use only circles larger than the 3-star equation clocks
        circles_to_use = [(x, y, r) for (x, y, r) in circles if r >= large_clock_threshold]
        
        # Area for 2-star: clocks are in upper-center area
        game_area_x_min = int(w * 0.2)
        game_area_x_max = int(w * 0.8)
        game_area_y_min = int(h * 0.1)
        game_area_y_max = int(h * 0.5)
        
    else:
        # Auto-detect mode (Keep original logic)
        radii = [r for (x, y, r) in circles]
        avg_radius = np.mean(radii)
        
        print(f"[Debug] Average radius: {avg_radius:.1f}px")
        
        large_circles = [(x, y, r) for (x, y, r) in circles if r >= large_clock_threshold]
        medium_circles = [(x, y, r) for (x, y, r) in circles if small_clock_threshold <= r < large_clock_threshold]
        
        print(f"[Debug] Large circles (r>={large_clock_threshold}): {len(large_circles)}")
        print(f"[Debug] Medium circles ({small_clock_threshold}<=r<{large_clock_threshold}): {len(medium_circles)}")
        
        if len(large_circles) >= 5:
            circles_to_use = large_circles
            print(f"[Debug] Using large circles only (2-star or 1-star suspected)")
            difficulty = 2 # Will be refined later
        elif len(medium_circles) >= 10:
            circles_to_use = medium_circles
            print(f"[Debug] Using medium circles only (3-star suspected)")
            difficulty = 3
        else:
            circles_to_use = large_circles + medium_circles
            print(f"[Debug] Using mixed circles")
            difficulty = 2 # Default fallback

        # Default area for auto-detect
        game_area_x_min = int(w * 0.15)
        game_area_x_max = int(w * 0.85)
        game_area_y_min = int(w * 0.1)
        game_area_y_max = int(h * 0.7)
    
    circles = circles_to_use
    
    # 3. Filter by 'is_clock_like' and Game Area
    valid_circles = []
    # Use the determined difficulty for is_clock_like filtering
    final_difficulty = force_difficulty if force_difficulty else difficulty 
    
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
            if is_clock_like(roi, difficulty=final_difficulty): 
                valid_circles.append((x, y, r))
    
    print(f"[Debug] Filtered to {len(valid_circles)} valid clock-like circles")
    
    if len(valid_circles) < 5:
        print(f"[Warning] Only found {len(valid_circles)} valid clock circles (need at least 5)")
        return None
    
    circles = valid_circles
    
    # 4. Split into Equation and Answer Clocks
    
    # Sort by Y position to find the vertical gap between equation and answer clocks
    circles_sorted = sorted(circles, key=lambda c: c[1])
    
    y_gaps = []
    for i in range(len(circles_sorted) - 1):
        gap = circles_sorted[i + 1][1] - circles_sorted[i][1]
        y_gaps.append((gap, i + 1))  # (gap_size, split_index)
    
    if y_gaps:
        y_gaps.sort(reverse=True)
        largest_gap_idx = y_gaps[0][1]
        
        equation_circles = circles_sorted[:largest_gap_idx]
        answer_circles = circles_sorted[largest_gap_idx:]
        
        if len(answer_circles) > 4:
            answer_circles = answer_circles[-4:] # Ensure we only use the bottom 4
        
        num_equation = len(equation_circles)
    else:
        answer_circles = circles_sorted[-4:]
        equation_circles = circles_sorted[:-4]
        num_equation = len(equation_circles)
    
    print(f"[Debug] Split into {num_equation} equation circles and {len(answer_circles)} answer circles")
    
    # 5. Final Difficulty and Count Determination
    
    if force_difficulty:
        difficulty = force_difficulty
    elif num_equation >= 10:
        difficulty = 3
    elif num_equation >= 2:
        difficulty = 2
    else:
        difficulty = 1
    
    if difficulty == 3:
        num_to_use = 15
    elif difficulty == 2:
        num_to_use = 2
    else:
        num_to_use = 1

    print(f"[Info] Detected {difficulty}-star difficulty ({num_to_use} equation clock{'s' if num_to_use > 1 else ''})")
    
    # Sort equation clocks by position (left-to-right, top-to-bottom)
    equation_circles = sorted(equation_circles, key=lambda c: (c[1] // 100, c[0]))
    
    # Sort answer clocks left-to-right
    answer_circles = sorted(answer_circles, key=lambda c: c[0])
    
    # Limit to number needed + 1 for result clock
    equation_circles = equation_circles[:num_to_use + 1]
    answer_circles = answer_circles[:4]
    
    # 6. Extract ROIs
    
    eq_rois = []
    for (x, y, r) in equation_circles:
        x, y, r = int(x), int(y), int(r)
        y1 = max(0, y - r)
        y2 = min(h, y + r)
        x1 = max(0, x - r)
        x2 = min(w, x + r)
        
        if y2 > y1 and x2 > x1:
            roi = frame[y1:y2, x1:x2]
            eq_rois.append(roi if roi.size > 0 else None)
        else:
            eq_rois.append(None)

    ans_rois = []
    for (x, y, r) in answer_circles:
        x, y, r = int(x), int(y), int(r)
        y1 = max(0, y - r)
        y2 = min(h, y + r)
        x1 = max(0, x - r)
        x2 = min(w, x + r)
        
        if y2 > y1 and x2 > x1:
            roi = frame[y1:y2, x1:x2]
            ans_rois.append(roi if roi.size > 0 else None)
        else:
            ans_rois.append(None)

    return eq_rois, ans_rois, difficulty