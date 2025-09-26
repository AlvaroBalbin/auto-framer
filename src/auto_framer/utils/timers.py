from __future__ import annotations
import time
from collections import deque

class FpsCalculator:
    """
    calculates fps in a lightweight manner
    OpenCV built-in functions are not too accurate
    call tick() once per frame, then returns the current smoothed FPS
    """
    def __init__(self, smoothing_window: int = 30): 
        self._times = deque(maxlen=smoothing_window) # throws old frames out after filled up
        self._latest = None # holds latest frame 
        self.fps = 0.0

    # collect time frames then place in a deque, afterwards get the avg of all fps frames(smoothing)
    def tick(self) -> float:
        now = time.perf_counter()
        if self._latest is not None:
            diff_time = now - self._latest
            if diff_time > 0:
                self._times.append(1/diff_time)
                self.fps = sum(self._times) / len(self._times)
        self._latest = now
        return self.fps
    
