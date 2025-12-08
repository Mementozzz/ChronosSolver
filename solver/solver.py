import time
import cv2
import platform
from .capture import capture_frame
from .detector import detect_clocks
from .clock_reader import read_clock
from .ui import find_best_answer

# Platform-specific imports for notifications
try:
    if platform.system() == 'Windows':
        from win10toast import ToastNotifier  # type: ignore
        TOAST_AVAILABLE = True
    else:
        TOAST_AVAILABLE = False
except ImportError:
    TOAST_AVAILABLE = False

# Try to import plyer for cross-platform notifications
try:
    import plyer  # type: ignore
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False

class ClockSolver:
    def __init__(self, verbose=False, show_window=True, enable_notifications=True):
        self.verbose = verbose
        self.show_window = show_window
        self.enable_notifications = enable_notifications
        self.window_name = "Chronos Solver - Press 'Q' or ESC to exit"
        self.paused = False
        self.last_frame = None
        self.last_result = None
        self.last_idx = None
        self.toaster = None
        
        # Initialize notification system
        if self.enable_notifications and platform.system() == 'Windows' and TOAST_AVAILABLE:
            try:
                self.toaster = ToastNotifier()
            except:
                self.log("[Warning] Could not initialize Windows notifications", 'debug')
        
    def log(self, message, level='info'):
        """Print message based on verbosity level"""
        if level == 'debug' and not self.verbose:
            return
        print(message)
        
    def send_notification(self, title, message):
        """Send a desktop notification"""
        if not self.enable_notifications:
            return
        
        try:
            if platform.system() == 'Windows':
                if self.toaster:
                    # Windows 10 Toast Notification
                    try:
                        self.toaster.show_toast(
                            title,
                            message,
                            duration=5,
                            threaded=True
                        )
                    except:
                        pass
                elif PLYER_AVAILABLE:
                    # Fallback to plyer
                    from plyer import notification as plyer_notify  # type: ignore
                    plyer_notify.notify(
                        title=title,
                        message=message,
                        app_name='Chronos Solver',
                        timeout=5
                    )
            elif PLYER_AVAILABLE:
                # Linux/Mac notifications via plyer
                from plyer import notification as plyer_notify  # type: ignore
                plyer_notify.notify(
                    title=title,
                    message=message,
                    app_name='Chronos Solver',
                    timeout=5
                )
        except Exception as e:
            self.log(f"[Warning] Could not send notification: {e}", 'debug')
    
    def focus_window(self):
        """Bring the OpenCV window to the foreground"""
        if not self.show_window:
            return
        
        try:
            # Try to set window to topmost
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            
            # On Windows, also try to use win32gui if available
            if platform.system() == 'Windows':
                try:
                    import win32gui  # type: ignore
                    import win32con  # type: ignore
                    hwnd = win32gui.FindWindow(None, self.window_name)
                    if hwnd:
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                        win32gui.SetForegroundWindow(hwnd)
                except ImportError:
                    pass
                except Exception as e:
                    self.log(f"[Debug] Could not focus window with win32gui: {e}", 'debug')
        except Exception as e:
            self.log(f"[Debug] Could not focus window: {e}", 'debug')
    
    def play_alert_sound(self):
        """Play a system alert sound"""
        try:
            if platform.system() == 'Windows':
                import winsound
                # Play a system sound
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            elif platform.system() == 'Darwin':  # macOS
                import os
                os.system('afplay /System/Library/Sounds/Glass.aiff')
            else:  # Linux
                import os
                os.system('paplay /usr/share/sounds/freedesktop/stereo/complete.oga 2>/dev/null || beep 2>/dev/null')
        except Exception as e:
            self.log(f"[Debug] Could not play alert sound: {e}", 'debug')
        """Print message based on verbosity level"""
        if level == 'debug' and not self.verbose:
            return
        print(message)

    def draw_debug_info(self, frame, eq_clocks, ans_clocks, result_time, result_idx):
        """Draw detection and result information on frame"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Add semi-transparent overlay for text background
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Add status text
        y_offset = 40
        cv2.putText(display_frame, "Chronos Clock Solver", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 30
        status_text = "PAUSED - Answer found!" if self.paused else "SCANNING..."
        status_color = (0, 255, 255) if self.paused else (255, 255, 255)
        cv2.putText(display_frame, f"Status: {status_text}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        y_offset += 25
        cv2.putText(display_frame, f"Clocks detected: {len(eq_clocks)} equation, {len(ans_clocks)} answers",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 30
        if result_time:
            time_str = f"Result: {result_time[0]:02d}:{result_time[1]:02d}"
            cv2.putText(display_frame, time_str, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y_offset += 35
        if result_idx is not None:
            answer_str = f">>> ANSWER: Option {result_idx + 1} <<<"
            cv2.putText(display_frame, answer_str, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add controls info
        controls_y = h - 50
        cv2.putText(display_frame, "Press 'Q' or ESC to exit", (20, controls_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display_frame, "Press 'R' to resume scanning", (20, controls_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display_frame

    def run(self):
        self.log("[ClockSolver] Starting Chronos Clock Solver...")
        self.log("[ClockSolver] Make sure the Chronos minigame is visible on screen!")
        
        if self.show_window:
            self.log("[ClockSolver] GUI window enabled. Press 'Q' or ESC to exit, 'R' to resume.")
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 720)
        else:
            self.log("[ClockSolver] Running in headless mode. Press CTRL+C to exit.")

        consecutive_failures = 0
        
        try:
            while True:
                # Handle paused state
                if self.paused:
                    if self.show_window and self.last_frame is not None:
                        # Continue showing the frozen frame with answer
                        display_frame = self.draw_debug_info(
                            self.last_frame, 
                            [], [], 
                            self.last_result, 
                            self.last_idx
                        )
                        cv2.imshow(self.window_name, display_frame)
                        
                        key = cv2.waitKey(100) & 0xFF
                        if key in [ord('q'), ord('Q'), 27]:  # 'q', 'Q', or ESC
                            self.log("\n[ClockSolver] Exit key pressed. Shutting down...")
                            break
                        elif key in [ord('r'), ord('R')]:  # 'R' to resume
                            self.log("\n[ClockSolver] Resuming scan...")
                            self.paused = False
                            self.last_frame = None
                            self.last_result = None
                            self.last_idx = None
                    else:
                        time.sleep(0.1)
                    continue

                # Normal scanning mode
                frame = capture_frame()

                circles = detect_clocks(frame)
                if circles is None:
                    consecutive_failures += 1
                    if consecutive_failures % 20 == 0:
                        self.log(f"[Warning] No clocks detected for {consecutive_failures} frames", 'debug')
                    
                    if self.show_window:
                        # Show frame even when no clocks detected
                        info_frame = frame.copy()
                        cv2.putText(info_frame, "Waiting for clocks...", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow(self.window_name, info_frame)
                        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:  # 'q', 'Q', or ESC
                            break
                    
                    time.sleep(0.1)
                    continue

                consecutive_failures = 0
                eq_clocks, ans_clocks = circles

                # Validate we have enough clocks
                if len(eq_clocks) < 11 or len(ans_clocks) < 4:
                    self.log(f"[Warning] Not enough clocks: {len(eq_clocks)} equation, {len(ans_clocks)} answers", 'debug')
                    
                    if self.show_window:
                        cv2.imshow(self.window_name, frame)
                        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
                            break
                    
                    time.sleep(0.1)
                    continue

                # Read the equation clocks (first 11, ignore final result clock)
                times = []
                valid_clocks = 0
                for i, roi in enumerate(eq_clocks[:-1]):
                    if roi is not None:
                        t = read_clock(roi)
                        times.append(t)
                        if t != (0, 0):
                            valid_clocks += 1
                        self.log(f"[Debug] Clock {i+1}: {t[0]:02d}:{t[1]:02d}", 'debug')
                    else:
                        times.append((0, 0))
                        self.log(f"[Debug] Clock {i+1}: Invalid ROI", 'debug')

                # Only proceed if we have enough valid clock readings
                if valid_clocks < 8:  # At least 8 out of 11 clocks should be readable
                    self.log(f"[Warning] Only {valid_clocks}/11 clocks readable, waiting for better frame", 'debug')
                    if self.show_window:
                        cv2.imshow(self.window_name, frame)
                        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
                            break
                    time.sleep(0.1)
                    continue

                # Compute sum of all times
                total_minutes = sum(h * 60 + m for h, m in times)
                total_hours = (total_minutes // 60) % 12
                total_min = total_minutes % 60

                self.log(f"[Result] Calculated time: {total_hours:02d}:{total_min:02d}")

                # Read answer choices
                answers = []
                valid_answers = 0
                for i, roi in enumerate(ans_clocks):
                    if roi is not None:
                        ans = read_clock(roi)
                        answers.append(ans)
                        if ans != (0, 0):
                            valid_answers += 1
                        self.log(f"[Debug] Answer {i+1}: {ans[0]:02d}:{ans[1]:02d}", 'debug')
                    else:
                        answers.append((0, 0))
                        self.log(f"[Debug] Answer {i+1}: Invalid ROI", 'debug')

                # Only accept result if we have all 4 valid answers
                if valid_answers < 4:
                    self.log(f"[Warning] Only {valid_answers}/4 answers readable, waiting for better frame", 'debug')
                    if self.show_window:
                        cv2.imshow(self.window_name, frame)
                        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
                            break
                    time.sleep(0.1)
                    continue

                idx = find_best_answer((total_hours, total_min), answers)
                if idx is not None:
                    answer_time = f"{answers[idx][0]:02d}:{answers[idx][1]:02d}"
                    self.log(f"✓ [ANSWER] Option {idx + 1} - {answer_time}")
                    self.log("=" * 50)
                    self.log("[ClockSolver] Answer found! Pausing capture.")
                    self.log("[ClockSolver] Press 'R' to resume for next puzzle, or 'Q' to exit.")
                    self.log("=" * 50)
                    
                    # Store the result and pause
                    self.last_result = (total_hours, total_min)
                    self.last_idx = idx
                    self.last_frame = frame.copy()
                    self.paused = True
                    
                    # Alert the user!
                    self.play_alert_sound()
                    self.send_notification(
                        "Chronos Answer Found! ✓",
                        f"Select Option {idx + 1} ({answer_time})"
                    )
                    if self.show_window:
                        self.focus_window()
                else:
                    self.log(f"[Warning] No exact match found for {total_hours:02d}:{total_min:02d}")

                # Display window with debug info
                if self.show_window:
                    display_frame = self.draw_debug_info(frame, eq_clocks, ans_clocks, 
                                                         self.last_result, self.last_idx)
                    cv2.imshow(self.window_name, display_frame)
                    
                    # Check for exit key
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('q'), ord('Q'), 27]:  # 'q', 'Q', or ESC
                        self.log("\n[ClockSolver] Exit key pressed. Shutting down...")
                        break

                time.sleep(0.05)
                
        except KeyboardInterrupt:
            self.log("\n[ClockSolver] Keyboard interrupt received. Exiting...")
        except Exception as e:
            self.log(f"[Error] Unexpected error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
        finally:
            if self.show_window:
                cv2.destroyAllWindows()
            self.log("[ClockSolver] Shutdown complete.")