import cv2
import numpy as np
import math

def read_clock(roi):
    # Validate ROI before processing
    if roi is None or roi.size == 0:
        return (0, 0)
    
    # Check if ROI has valid dimensions
    if roi.shape[0] < 10 or roi.shape[1] < 10:
        return (0, 0)
    
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        print(f"[Warning] cvtColor failed: {e}")
        return (0, 0)
    
    h, w = gray.shape
    center = (w // 2, h // 2)
    
    # Create a mask for the center area (ignore outer tick marks)
    mask = np.zeros_like(gray)
    inner_radius = int(min(h, w) * 0.15)
    outer_radius = int(min(h, w) * 0.45)
    cv2.circle(mask, center, outer_radius, 255, -1)
    cv2.circle(mask, center, inner_radius, 0, -1)
    
    # Try to isolate the clock hands
    # Method 1: Look for dark hands on light background
    _, thresh_dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    thresh_dark = cv2.bitwise_and(thresh_dark, mask)
    
    # Method 2: Look for light hands on dark background  
    _, thresh_light = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    thresh_light = cv2.bitwise_and(thresh_light, mask)
    
    # Choose the threshold with more content (hands)
    if np.sum(thresh_dark) > np.sum(thresh_light):
        thresh = thresh_dark
    else:
        thresh = thresh_light
    
    # Find contours (clock hands)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return (0, 0)
    
    # Analyze each contour to find clock hands
    hands = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:  # Too small to be a hand
            continue
        
        # Get the farthest point from center
        max_dist = 0
        farthest_point = None
        
        for point in contour:
            pt = point[0]
            dist = math.hypot(pt[0] - center[0], pt[1] - center[1])
            if dist > max_dist:
                max_dist = dist
                farthest_point = pt
        
        if farthest_point is None or max_dist < min(h, w) * 0.2:
            continue
        
        # Calculate angle to farthest point
        dx = farthest_point[0] - center[0]
        dy = center[1] - farthest_point[1]  # Invert Y
        
        angle = math.degrees(math.atan2(dx, dy))
        angle = (angle + 360) % 360
        
        hands.append({
            'angle': angle,
            'length': max_dist,
            'area': area
        })
    
    # If contour method didn't work well, fall back to line detection
    if len(hands) < 2:
        lines = cv2.HoughLinesP(
            thresh, 1, np.pi/180,
            threshold=int(min(h, w) * 0.15),
            minLineLength=int(min(h, w) * 0.3),
            maxLineGap=int(min(h, w) * 0.2)
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate which endpoint is farther from center
                dist1 = math.hypot(x1 - center[0], y1 - center[1])
                dist2 = math.hypot(x2 - center[0], y2 - center[1])
                
                if dist1 > dist2:
                    end_x, end_y = x1, y1
                    length = dist1
                else:
                    end_x, end_y = x2, y2
                    length = dist2
                
                if length < min(h, w) * 0.25:
                    continue
                
                dx = end_x - center[0]
                dy = center[1] - end_y
                
                angle = math.degrees(math.atan2(dx, dy))
                angle = (angle + 360) % 360
                
                hands.append({
                    'angle': angle,
                    'length': length,
                    'area': 100  # Default area for lines
                })
    
    if len(hands) < 1:
        return (0, 0)
    
    # Sort by length - longest is likely minute hand
    hands.sort(key=lambda h: h['length'], reverse=True)
    
    # Take top 2 hands
    if len(hands) == 1:
        minute_angle = hands[0]['angle']
        hour_angle = hands[0]['angle']
    else:
        # The longer hand is minute, shorter is hour
        minute_angle = hands[0]['angle']
        hour_angle = hands[1]['angle']
    
    # Convert angles to time
    minute = round(minute_angle / 6) % 60
    hour = round(hour_angle / 30) % 12
    
    # Special case: if hour and minute are very close, might have detected same hand twice
    if abs(minute_angle - hour_angle) < 15 and len(hands) > 2:
        # Try the third hand
        hour_angle = hands[2]['angle']
        hour = round(hour_angle / 30) % 12
    
    return hour, minute