"""
Test script to run the clock solver on a static image file
Usage: python test_image.py <image_path> [--difficulty 2|3] [--brightness -50]
Example: python test_image.py chronos.png -d 3 -b -50
"""
import sys
import cv2
import argparse
import numpy as np
from detector import detect_clocks
from clock_reader import read_clock
from ui import find_best_answer

# Exposure Adjustment Function
def adjust_exposure(frame, brightness_offset=-50):
    """
    Adjusts the brightness of the frame. A negative offset darkens the image.
    This helps increase contrast for detection on bright game backgrounds.
    """
    if frame is None or frame.size == 0:
        return frame
    # Use cv2.convertScaleAbs for a simple linear adjustment: O = alpha*I + beta
    # alpha=1.0 for no contrast change, beta=brightness_offset (e.g., -50)
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness_offset)
    return adjusted_frame


def test_image(image_path, difficulty=None, brightness_offset=-50):
    print(f"[Test] Loading image: {image_path}")
    
    # Load the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[Error] Could not load image: {image_path}")
        print("[Info] Make sure the file path is correct and the image is a valid format (jpg, png, etc.)")
        return

    h, w = frame.shape[:2]

    print(f"[Test] Image loaded: {w}x{h} pixels")
    
    # Apply exposure adjustment
    print(f"[Test] Applying brightness adjustment: {brightness_offset}")
    frame_for_detection = adjust_exposure(frame, brightness_offset)
    
    # Detect clocks - use the adjusted frame
    print("[Test] Detecting clocks...")
    result = detect_clocks(frame_for_detection, force_difficulty=difficulty)
    
    if result is None:
        print("[Error] No clocks detected in the image!")
        print("[Info] Make sure the Chronos puzzle is clearly visible and not obscured")
        
        # Save debug image showing what was detected
        gray = cv2.cvtColor(frame_for_detection, cv2.COLOR_BGR2GRAY) # Use adjusted frame
        blur = cv2.GaussianBlur(gray, (7, 7), 1)
        edges = cv2.Canny(blur, 50, 150)
        cv2.imwrite('debug_edges.png', edges)
        print("[Debug] Saved edge detection to 'debug_edges.png'")
        return
    
    eq_clocks, ans_clocks, difficulty = result
    print(f"[Test] Found {len(eq_clocks)} equation clocks and {len(ans_clocks)} answer clocks")
    print(f"[Test] Difficulty: {difficulty}-star")
    
    # Determine how many clocks to read
    if difficulty == 1:
        num_to_read = 1
    elif difficulty == 2:
        num_to_read = 2
    else:
        num_to_read = 15
    
    # Read equation clocks
    print("\n[Test] Reading equation clocks:")
    times = []
    for i, roi in enumerate(eq_clocks[:num_to_read]):
        if roi is not None:
            t = read_clock(roi)
            # Fix hour wrapping for display: 0 should be 12
            h_time = 12 if t[0] == 0 else t[0]
            times.append(t)
            print(f"  Clock {i+1}: {h_time:02d}:{t[1]:02d}")
        else:
            times.append((0, 0))
            print(f"  Clock {i+1}: Invalid ROI")
    
    # Calculate result
    print(f"\n[Result] {difficulty}-star difficulty")
    
    if difficulty == 1:
        total_hours = times[0][0]
        total_min = times[0][1]
    else:
        total_minutes = sum(h_val * 60 + m for h_val, m in times)
        total_hours = (total_minutes // 60) % 12
        total_min = total_minutes % 60
    
    # Fix hour wrapping for final result: 0 should be 12
    if total_hours == 0: total_hours = 12
        
    print(f"[Result] Calculated time: {total_hours:02d}:{total_min:02d}")
    
    # Read answer choices
    print("\n[Test] Reading answer choices:")
    answers = []
    for i, roi in enumerate(ans_clocks):
        if roi is not None:
            ans = read_clock(roi)
            answers.append(ans)
            h_ans = 12 if ans[0] == 0 else ans[0]
            print(f"  Answer {i+1}: {h_ans:02d}:{ans[1]:02d}")
        else:
            answers.append((0, 0))
            print(f"  Answer {i+1}: Invalid ROI")
    
    # Find best match
    result = find_best_answer((total_hours, total_min), answers)
    
    print("\n" + "=" * 50)
    if result is not None and result[0] is not None:
        idx, time_diff = result
        answer_time = answers[idx]
        h_ans_final = 12 if answer_time[0] == 0 else answer_time[0]
        
        if time_diff == 0:
            print(f"✓ [ANSWER] Select Option {idx + 1} ({h_ans_final:02d}:{answer_time[1]:02d}) - EXACT MATCH")
        else:
            print(f"✓ [ANSWER] Select Option {idx + 1} ({h_ans_final:02d}:{answer_time[1]:02d}) - Off by {time_diff} minute(s)")
    else:
        print(f"✗ [NO MATCH] Could not find match within 5 minutes for {total_hours:02d}:{total_min:02d}")
        h_answers = [(12 if h_val == 0 else h_val, m) for h_val, m in answers]
        print(f"  Available answers: {h_answers}")
    print("=" * 50)
    
    # Draw debug visualization on the adjusted frame
    debug_frame = frame_for_detection.copy()
    
    # Draw detected circles (for debugging)
    # Note: We rely on the raw Hough Circles for visualization here
    gray = cv2.cvtColor(frame_for_detection, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (7, 7), 1)
    edges = cv2.Canny(blur, 50, 150)
    
    # Use the same general parameters as the detector's initial call (W IS NOW DEFINED)
    if w >= 2560:
        hough_min_dist = 100
        hough_min_radius = 50
        hough_max_radius = 180
    elif w >= 1920:
        hough_min_dist = 80
        hough_min_radius = 40
        hough_max_radius = 140
    else:
        hough_min_dist = 70
        hough_min_radius = 35
        hough_max_radius = 110
        
    circles_raw = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=hough_min_dist,
        param1=80, param2=25, minRadius=hough_min_radius, maxRadius=hough_max_radius
    )
    
    if circles_raw is not None:
        circles_raw = np.uint16(np.around(circles_raw))[0]
        # Only draw the circles that were actually processed (up to 20 for visual clarity)
        circles_to_draw = circles_raw[:20]
        
        for (x, y, r) in circles_to_draw: 
            # Check if circle was classified as equation (small/medium) or answer (large/low)
            color = (0, 255, 0) # Default Green (Equation)
            # Use W, which is now defined
            if r > 90 and w >= 1920 and y > frame_for_detection.shape[0] * 0.6:
                 color = (0, 255, 255) # Cyan for probable large Answer
            elif r > 120 and w >= 2560 and y > frame_for_detection.shape[0] * 0.6:
                 color = (0, 255, 255) # Cyan for probable large Answer
                 
            cv2.circle(debug_frame, (int(x), int(y)), int(r), color, 2)

    # Add result text
    if result is not None and result[0] is not None:
        idx = result[0]
        cv2.putText(debug_frame, f"ANSWER: Option {idx + 1}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    # Save debug image
    cv2.imwrite('debug_result.png', debug_frame)
    print("\n[Debug] Saved visualization to 'debug_result.png'")
    
    # Show the image
    cv2.imshow('Chronos Solver - Test Result', debug_frame)
    print("\n[Info] Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Chronos Clock Solver on a static image')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('-d', '--difficulty', type=int, choices=[2, 3], 
                        help='Force difficulty level (2 or 3 star)')
    parser.add_argument('-b', '--brightness', type=int, default=-50, 
                        help='Brightness offset for exposure adjustment (e.g., -50)')
    
    args = parser.parse_args()
    
    if args.difficulty:
        print(f"[Info] Forcing {args.difficulty}-star difficulty mode")
    
    test_image(args.image_path, difficulty=args.difficulty, brightness_offset=args.brightness)