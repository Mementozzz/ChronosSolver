import time
from .capture import capture_frame
from .detector import detect_clocks
from .clock_reader import read_clock
from .ui import find_best_answer

class ClockSolver:

    def run(self):
        print("[ClockSolver] Running... Press CTRL+C to exit.")
        print("[ClockSolver] Make sure the Chronos minigame is visible on screen!")

        consecutive_failures = 0
        
        while True:
            try:
                frame = capture_frame()

                circles = detect_clocks(frame)
                if circles is None:
                    consecutive_failures += 1
                    if consecutive_failures % 20 == 0:
                        print(f"[Warning] No clocks detected for {consecutive_failures} frames")
                    time.sleep(0.1)
                    continue

                consecutive_failures = 0  # Reset on success
                eq_clocks, ans_clocks = circles

                # Validate we have enough clocks
                if len(eq_clocks) < 11 or len(ans_clocks) < 4:
                    print(f"[Warning] Not enough clocks detected: {len(eq_clocks)} equation, {len(ans_clocks)} answers")
                    time.sleep(0.1)
                    continue

                # Read the equation clocks
                times = []
                for i, roi in enumerate(eq_clocks[:-1]):
                    if roi is not None:
                        t = read_clock(roi)
                        times.append(t)
                    else:
                        times.append((0, 0))

                # Compute sum of all times
                total_minutes = sum(h * 60 + m for h, m in times)
                total_hours = (total_minutes // 60) % 12
                total_min = total_minutes % 60

                print(f"[Debug] Calculated time: {total_hours:02d}:{total_min:02d}")

                # Read answer choices
                answers = []
                for roi in ans_clocks:
                    if roi is not None:
                        answers.append(read_clock(roi))
                    else:
                        answers.append((0, 0))

                print(f"[Debug] Answer choices: {answers}")

                idx = find_best_answer((total_hours, total_min), answers)
                if idx is not None:
                    print(f"✓ [Solver] Best match = Option {idx + 1}")
                    print("-" * 50)
                else:
                    print(f"[Warning] No exact match found for {total_hours:02d}:{total_min:02d}")

                time.sleep(0.05)
                
            except KeyboardInterrupt:
                print("\n[ClockSolver] Exiting...")
                break
            except Exception as e:
                print(f"[Error] Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)