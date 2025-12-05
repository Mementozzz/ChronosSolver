"""
Auto solver for the Chronos minigame.
Cross-platform: Windows / macOS / Linux.

Requirements:
    pip install opencv-python numpy mss pyautogui

Usage: run the script while the game is visible on screen.
Set AUTO_CLICK = True to enable clicking the screen on the chosen answer.
Set DEBUG = True to show intermediate windows for tuning (slower).
"""

import cv2
import numpy as np
import mss
import math
import time
import statistics
import pyautogui

# -----------------------
# Config
# -----------------------
DEBUG = False          # set True to see debugging windows
AUTO_CLICK = False     # set True to auto-click chosen answer
FRAME_INTERVAL = 0.12  # seconds between frames (0.12 ≈ 8 FPS; lower for more load)

# -----------------------
# Hough / contour parameters (auto-scaled by screen diagonal)
MIN_CLOCK_RADIUS_RATIO = 0.03  # min radius relative to screen diagonal
MAX_CLOCK_RADIUS_RATIO = 0.12  # max radius relative to screen diagonal

# Filtering of circular contours
MIN_CIRCULARITY = 0.5    # 0..1, 1 is perfect circle
MIN_CONTOUR_AREA_RATIO = 0.002  # contour area relative to screen area (to filter tiny noise)

# HoughLinesP parameters (will be scaled by clock radius)
HOUGH_THRESH_BASE = 30
MIN_LINE_LEN_FACTOR = 0.35
MAX_LINE_GAP_FACTOR = 12
# -----------------------

# -----------------------
# Utilities
# -----------------------
def screen_size():
    with mss.mss() as sct:
        mon = sct.monitors[0]  # whole virtual screen
        return mon['width'], mon['height']

def grab_frame():
    with mss.mss() as sct:
        mon = sct.monitors[0]
        sct_img = sct.grab(mon)
        img = np.array(sct_img)[:, :, :3].copy()
        return img

def draw_debug(name, img, scale=1.0):
    if DEBUG:
        cv2.imshow(name, img if scale == 1.0 else cv2.resize(img, (0,0), fx=scale, fy=scale))

# -----------------------
# Detect circular clock outlines robustly (contours + circularity)
# returns list of (cx, cy, r)
# -----------------------
def detect_clock_circles(frame):
    h, w = frame.shape[:2]
    screen_diag = math.hypot(w, h)
    min_r = int(screen_diag * MIN_CLOCK_RADIUS_RATIO)
    max_r = int(screen_diag * MAX_CLOCK_RADIUS_RATIO)

    # Convert to gray, strong blur to ignore background texture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    # Contrast enhancement: CLAHE to mitigate bright glare
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blur)

    # Canny edges (thresholds auto via median)
    v = np.median(enhanced)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(enhanced, lower, upper)

    # Dilate slightly to close gaps on ring stroke
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours on edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    screen_area = w * h
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < screen_area * MIN_CONTOUR_AREA_RATIO:
            continue

        # Compute circularity: 4*pi*area / perimeter^2
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circularity = 4 * math.pi * area / (peri * peri)

        # Fit enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        radius = int(radius)

        if radius < min_r or radius > max_r:
            continue
        if circularity < MIN_CIRCULARITY:
            continue

        # Also check that the contour approximates a ring (inner hole)
        # compute mean intensity inside the circle and outside near ring to ensure white stroke
        candidates.append((int(x), int(y), radius, circularity, area))

    # If many candidates, keep the largest by radius and area heuristics
    # Remove duplicates (close centers)
    filtered = []
    candidates.sort(key=lambda c: (c[2], c[4]), reverse=True)
    for cand in candidates:
        x, y, r, circ, area = cand
        too_close = False
        for fx, fy, fr in filtered:
            if math.hypot(fx - x, fy - y) < fr * 0.6:
                too_close = True
                break
        if not too_close:
            filtered.append((x, y, r))

    return filtered

# -----------------------
# Cluster detected circles into top grid(12) and bottom answers(4).
# Returns: eq_circles_sorted (12 entries left-to-right, top-to-bottom),
#          ans_circles_sorted (4 entries left-to-right)
# -----------------------
def cluster_and_sort_circles(circles):
    if len(circles) < 15:
        # sometimes glare hides outermost ring; still continue but may fail
        pass

    # sort by y ascending
    circles_sorted = sorted(circles, key=lambda c: c[1])

    # We expect 3 rows on top and 1 row bottom (4 answers).
    # Use KMeans-like split by y to find 4 clusters (3 small + 1 bottom)
    ys = np.array([c[1] for c in circles_sorted]).reshape(-1,1)
    # simple agglomerative clustering: group by gaps in sorted y
    ys_sorted = sorted(ys.flatten())
    gaps = np.diff(ys_sorted)
    # big gap indicates separation between top grid and answers
    if len(gaps) == 0:
        return [], []

    # find largest gap
    gap_idx = int(np.argmax(gaps))
    split_value = (ys_sorted[gap_idx] + ys_sorted[gap_idx + 1]) / 2.0

    top = [c for c in circles_sorted if c[1] <= split_value]
    bottom = [c for c in circles_sorted if c[1] > split_value]

    # If bottom not 4 circles, attempt alternative: take lowest 4 as bottom
    if len(bottom) != 4:
        # fallback: take 4 lowest centers
        bottom = sorted(circles_sorted, key=lambda c: c[1])[-4:]
        top = [c for c in circles_sorted if c not in bottom]

    # At this point we expect top to have about 12 circles
    # If top has more, pick the topmost 12 by y
    top = sorted(top, key=lambda c: (c[1], c[0]))
    if len(top) > 12:
        top = top[:12]

    # Sort top into 3 rows x 4 columns:
    # First cluster into 3 rows by y
    top_rows = sorted(top, key=lambda c: c[1])
    # split into 3 groups by y
    rows = []
    if len(top_rows) >= 12:
        # group into 3 equal groups
        for i in range(3):
            group = top_rows[i*4:(i+1)*4]
            rows.append(sorted(group, key=lambda c: c[0]))  # left to right
    else:
        # fallback: cluster by y gaps
        ys_top = [c[1] for c in top_rows]
        breaks = np.array(ys_top)
        # naive partition: divide into 3 groups
        k = len(top_rows)
        per_row = max(1, k // 3)
        rows = []
        for i in range(3):
            slice_ = top_rows[i*per_row: (i+1)*per_row]
            rows.append(sorted(slice_, key=lambda c: c[0]))

        # flatten to 12 by padding if needed
    # flatten rows
    eq_circles_sorted = []
    for r in rows:
        eq_circles_sorted.extend(r)

    # Sort bottom answers left-to-right
    ans_circles_sorted = sorted(bottom, key=lambda c: c[0])

    return eq_circles_sorted, ans_circles_sorted

# -----------------------
# Read time from a clock ROI (cx,cy,r)
# Returns None or (hour, minute)
# -----------------------
def read_clock_time(frame, cx, cy, radius):
    # Extract square ROI around circle
    r = int(radius * 0.9)
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(frame.shape[1], cx + r)
    y2 = min(frame.shape[0], cy + r)
    roi = frame[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return None

    # Preprocess ROI to isolate white hands (white -> high V, low-medium S)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # White mask: low saturation and high value. Thresholds tuned for bright backgrounds.
    # Use adaptive thresholds: value threshold relative to ROI median
    v_med = int(np.median(v))
    value_thresh = max(200, v_med + 45)  # prefer very bright pixels
    sat_thresh = 90                       # low saturation for white-ish strokes

    _, v_mask = cv2.threshold(v, value_thresh, 255, cv2.THRESH_BINARY)
    _, s_mask = cv2.threshold(s, sat_thresh, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(v_mask, s_mask)

    # Remove the outer ring by eroding slightly, leaving hands
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Optionally thin
    mask = cv2.ximgproc.thinning(mask) if hasattr(cv2.ximgproc, 'thinning') else mask

    # Edge detect on masked ROI
    edges = cv2.Canny(mask, 50, 150)

    # HoughLinesP to detect lines
    min_line_len = int(radius * MIN_LINE_LEN_FACTOR)
    max_line_gap = int(radius * MAX_LINE_GAP_FACTOR)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=HOUGH_THRESH_BASE,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:
        # fallback: try edges on ROI gray
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges2 = cv2.Canny(gray, 30, 120)
        lines = cv2.HoughLinesP(edges2, 1, np.pi/180, threshold=HOUGH_THRESH_BASE,
                                minLineLength=min_line_len//2, maxLineGap=max_line_gap)

    if lines is None:
        return None

    # Pick up to 4 strong lines, compute lengths & angles
    processed = []
    for l in lines:
        x1l, y1l, x2l, y2l = l[0]
        length = math.hypot(x2l - x1l, y2l - y1l)
        # compute center distance to ROI center to prefer lines passing near center
        cx_local = (roi.shape[1]) / 2.0
        cy_local = (roi.shape[0]) / 2.0
        midx = (x1l + x2l) / 2.0
        midy = (y1l + y2l) / 2.0
        dist_center = math.hypot(midx - cx_local, midy - cy_local)
        score = length - dist_center  # prefer long lines close to center
        processed.append((score, length, (x1l, y1l, x2l, y2l)))

    processed.sort(reverse=True)
    # keep top 2 lines (minute and hour usually)
    top_lines = [p[2] for p in processed[:3]]  # allow 3 to be robust

    # Convert each line to angle relative to upward (12 o'clock)
    def line_angle_deg(x1, y1, x2, y2):
        # angle where 0 is pointing up (12 o'clock), increase clockwise
        dx = x2 - x1
        dy = y1 - y2   # flip y so upward is positive
        ang = math.degrees(math.atan2(dy, dx))
        ang = (ang + 360) % 360
        # Convert so that 0 is at 12 o'clock: atan2 returns 0 at x+ axis; rotate by 90
        ang = (ang + 90) % 360
        return ang

    line_infos = []
    for (x1l, y1l, x2l, y2l) in top_lines:
        ang = line_angle_deg(x1l, y1l, x2l, y2l)
        length = math.hypot(x2l-x1l, y2l-y1l)
        line_infos.append((length, ang))

    if len(line_infos) < 1:
        return None

    # sort by length -> longest is minute, shorter is hour (approx)
    line_infos.sort(reverse=True)  # longest first
    minute_angle = line_infos[0][1]
    if len(line_infos) >= 2:
        hour_angle = line_infos[1][1]
    else:
        # fallback: approximate hour from minute (minute/12)
        hour_angle = minute_angle / 12.0

    # convert angles to minute/hour values
    minute = int(round(minute_angle / 6.0)) % 60
    hour = int(round(hour_angle / 30.0)) % 12

    # Extra sanity: if minute is near 0 but hour hand angle suggests e.g., 3:30, adjust hour
    # compute hour fractional from hour_angle
    hour_frac = (hour_angle % 360) / 30.0
    hour_floor = int(math.floor(hour_frac)) % 12
    hour = hour_floor

    return hour, minute

# -----------------------
# Solve puzzle in one frame
# -----------------------
def solve_once(frame):
    circles = detect_clock_circles(frame)
    if not circles:
        return None

    eq_circles, ans_circles = cluster_and_sort_circles(circles)

    # If we don't have expected counts, bail (or continue best-effort)
    if len(eq_circles) < 11 or len(ans_circles) < 4:
        # best-effort: try taking top 12 of circles as eq and bottom 4 as ans
        circles_sorted = sorted(circles, key=lambda c: (c[1], c[0]))
        if len(circles_sorted) >= 16:
            eq_circles = circles_sorted[:12]
            ans_circles = circles_sorted[12:16]
        else:
            return None

    # read times
    times = []
    for (cx, cy, r) in eq_circles:
        t = read_clock_time(frame, cx, cy, r)
        if t is None:
            # if any read fails, abort this frame
            return None
        times.append(t)

    # we interpret puzzle as sum of first 11 clocks; the 12th (far right) is the displayed result (ignored)
    if len(times) >= 12:
        equation_times = times[:11]
    else:
        equation_times = times[:-1]

    # convert times to minute counts (0..719)
    def to_minutes(h, m):
        return (h % 12) * 60 + (m % 60)

    total = sum(to_minutes(h, m) for (h, m) in equation_times) % (12 * 60)

    # read answers and match
    ans_vals = []
    for (cx, cy, r) in ans_circles:
        t = read_clock_time(frame, cx, cy, r)
        if t is None:
            ans_vals.append(None)
        else:
            ans_vals.append(to_minutes(*t))

    # pick index where ans == total
    chosen_idx = None
    for i, v in enumerate(ans_vals):
        if v == total:
            chosen_idx = i
            break

    # debug drawing
    if DEBUG:
        dbg = frame.copy()
        for (cx, cy, r) in eq_circles:
            cv2.circle(dbg, (cx, cy), r, (0,255,0), 2)
        for (cx, cy, r) in ans_circles:
            cv2.circle(dbg, (cx, cy), r, (255,0,0), 2)
        cv2.putText(dbg, f"Total minutes: {total}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        draw_debug("debug", dbg, scale=0.7)

    return {
        "total_minutes": total,
        "answers": ans_vals,
        "chosen_index": chosen_idx,
        "eq_circles": eq_circles,
        "ans_circles": ans_circles
    }

# -----------------------
# Auto-click helper
# -----------------------
def click_answer_on_screen(ans_circle):
    cx, cy, r = ans_circle
    # pyautogui uses screen coordinates; we detect circles using full-screen capture,
    # so coordinates map directly (no translation required)
    x = cx
    y = cy
    pyautogui.moveTo(x, y, duration=0.05)
    pyautogui.click()

# -----------------------
# Main loop
# -----------------------
def main_loop():
    print("Starting clock solver. DEBUG:", DEBUG, "AUTO_CLICK:", AUTO_CLICK)
    last_print = 0.0
    while True:
        frame = grab_frame()
        start = time.time()
        result = solve_once(frame)
        elapsed = time.time() - start

        if result:
            total = result['total_minutes']
            ans_vals = result['answers']
            idx = result['chosen_index']
            # convert total to h:m
            h = (total // 60) % 12
            m = total % 60
            now = time.time()
            # print at most 1 per 0.5s to avoid floods
            if now - last_print > 0.5:
                print(f"[{elapsed*1000:.0f} ms] Computed result: {h:02d}:{m:02d}  answers={ans_vals} chosen={idx}")
                last_print = now

            if idx is not None and AUTO_CLICK:
                # click the corresponding answer circle
                ans_circle = result['ans_circles'][idx]
                click_answer_on_screen(ans_circle)
                # short pause to avoid double-clicks
                time.sleep(0.25)
        else:
            # occasional log for debugging
            now = time.time()
            if now - last_print > 2.0:
                print(f"[{elapsed*1000:.0f} ms] No reliable detection this frame...")
                last_print = now

        # basic key handling for debug windows
        if DEBUG and cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(FRAME_INTERVAL)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("Exiting.")
    finally:
        cv2.destroyAllWindows()
