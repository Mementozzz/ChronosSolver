import time
from .capture import capture_frame
from .detector import detect_clocks
from .clock_reader import read_clock
from .ui import find_best_answer

class ClockSolver:

    def run(self):
        print("[ClockSolver] Running... Press CTRL+C to exit.")

        while True:
            frame = capture_frame()

            circles = detect_clocks(frame)
            if circles is None:
                continue

            eq_clocks, ans_clocks = circles

            # Read the equation clocks (first 11)
            times = []
            for roi in eq_clocks[:-1]:  # ignore final result clock
                t = read_clock(roi)
                if t is not None:
                    times.append(t)
                else:
                    times.append((0, 0))

            # Compute sum of all times
            total_minutes = sum(h * 60 + m for h, m in times)
            total_hours = (total_minutes // 60) % 12
            total_min = total_minutes % 60

            # Read answer choices
            answers = [read_clock(a) for a in ans_clocks]

            idx = find_best_answer((total_hours, total_min), answers)
            if idx is not None:
                print(f"[Solver] Best match = option {idx+1}")

            time.sleep(0.05)