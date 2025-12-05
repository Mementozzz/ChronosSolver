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
    
    thres = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 9
    )

    lines = cv2.HoughLinesP(
        thres, 1, np.pi/180,
        threshold=20,
        minLineLength=20,
        maxLineGap=5
    )

    if lines is None:
        return (0, 0)

    center = (roi.shape[1]//2, roi.shape[0]//2)

    hour_angle = None
    minute_angle = None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y1 - y2
        ang = math.degrees(math.atan2(dy, dx))
        ang = (ang + 360) % 360

        length = math.hypot(dx, dy)

        if length > 40:
            minute_angle = ang
        else:
            hour_angle = ang

    if minute_angle is None or hour_angle is None:
        return (0, 0)

    minute = round(minute_angle / 6) % 60
    hour = round(hour_angle / 30) % 12

    return hour, minute