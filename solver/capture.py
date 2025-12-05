import numpy as np
import mss

def capture_frame():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = np.array(sct.grab(monitor))[:, :, :3]
        return img