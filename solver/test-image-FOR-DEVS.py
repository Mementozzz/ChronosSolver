"""
Test script to run the clock solver on a static image file
Usage: python test-image-FOR-DEVS.py <image_path>
"""
import sys
import cv2
from solver.detector import detect_clocks
from solver.clock_reader import read_clock
from solver.ui import find_best_answer

def test_image(image_path):
    print(f"[Test] Loading image: {image_path}")
    
    # Load the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[Error] Could not load image: {image_path}")
        print("[Info] Make sure the file path is correct and the image is a valid format (jpg, png, etc.)")
        return
    
    print(f"[Test] Image loaded: {frame.shape[1]}x{frame.shape[0]} pixels")
    
    # Detect clocks
    print("[Test] Detecting clocks...")
    circles = detect_clocks(frame)
    
    if circles is None:
        print("[Error] No clocks detected in the image!")
        print("[Info] Make sure the Chronos puzzle is clearly visible and not obscured")
        
        # Save debug image showing what was detected
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 1)
        edges = cv2.Canny(blur, 50, 150)
        cv2.imwrite('debug_edges.png', edges)
        print("[Debug] Saved edge detection to 'debug_edges.png'")
        return
    
    eq_clocks, ans_clocks = circles
    print(f"[Test] Found {len(eq_clocks)} equation clocks and {len(ans_clocks)} answer clocks")
    
    # Read equation clocks
    print("\n[Test] Reading equation clocks:")
    times = []
    for i, roi in enumerate(eq_clocks[:-1]):  # Skip the result clock
        if roi is not None:
            t = read_clock(roi)
            times.append(t)
            print(f"  Clock {i+1}: {t[0]:02d}:{t[1]:02d}")
        else:
            times.append((0, 0))
            print(f"  Clock {i+1}: Invalid ROI")
    
    # Calculate result
    total_minutes = sum(h * 60 + m for h, m in times)
    total_hours = (total_minutes // 60) % 12
    total_min = total_minutes % 60
    
    print(f"\n[Result] Calculated time: {total_hours:02d}:{total_min:02d}")
    
    # Read answer choices
    print("\n[Test] Reading answer choices:")
    answers = []
    for i, roi in enumerate(ans_clocks):
        if roi is not None:
            ans = read_clock(roi)
            answers.append(ans)
            print(f"  Answer {i+1}: {ans[0]:02d}:{ans[1]:02d}")
        else:
            answers.append((0, 0))
            print(f"  Answer {i+1}: Invalid ROI")
    
    # Find best match
    idx = find_best_answer((total_hours, total_min), answers)
    
    print("\n" + "=" * 50)
    if idx is not None:
        print(f"✓ [ANSWER] Select Option {idx + 1} ({answers[idx][0]:02d}:{answers[idx][1]:02d})")
    else:
        print(f"✗ [NO MATCH] Could not find exact match for {total_hours:02d}:{total_min:02d}")
        print(f"  Available answers: {answers}")
    print("=" * 50)
    
    # Draw debug visualization
    debug_frame = frame.copy()
    
    # Draw detected circles (for debugging)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1)
    edges = cv2.Canny(blur, 50, 150)
    circles_raw = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=70,
        param1=100, param2=25, minRadius=35, maxRadius=120
    )
    
    if circles_raw is not None:
        import numpy as np
        circles_raw = np.uint16(np.around(circles_raw))[0]
        for i, (x, y, r) in enumerate(circles_raw[:16]):
            color = (0, 255, 0) if i < 12 else (0, 255, 255)  # Green for equation, cyan for answers
            cv2.circle(debug_frame, (int(x), int(y)), int(r), color, 2)
            cv2.putText(debug_frame, str(i+1), (int(x)-10, int(y)+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add result text
    if idx is not None:
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
    if len(sys.argv) < 2:
        print("Usage: python test_image.py <image_path>")
        print("Example: python test_image.py chronos_screenshot.png")
        sys.exit(1)
    
    test_image(sys.argv[1])