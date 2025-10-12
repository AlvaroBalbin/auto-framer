from __future__ import annotations
import cv2 as cv
import numpy as np
import pyvirtualcam

try:
    import pyvirtualcam
    _HAS_VCAM = True
except Exception:
    _HAS_VCAM = False

class Renderer:
    """
    - displays preview, shows the live video in an OpenCV window 
    - it also gives the output into a virtual webcam, so it can 
    stream on zoom/meets/teams
    """
    def __init__(self, width: int, height: int, fps: int = 30, enable_virtualcam: bool = False):
        self.width = width
        self.height = height
        self.enable_virtual_camera = bool(enable_virtualcam)
        self.fps = fps
        self.vcam = None

    # two function that get called with context managers
    def __enter__(self):
        if self.enable_virtual_camera and _HAS_VCAM and self.vcam is None:
            self.vcam = pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps, backend="obs")
            print(f"[Renderer] Virtual Camera Enabled: {self.vcam.device}")
        elif self.enable_virtual_camera and _HAS_VCAM is None:
            print("[Renderer] pyvirtualcam not installed will now disable virtual camera output.")
        else:
            print(f"[Renderer] Virtual Camera not available, disabling it")
            self.vcam = None
        return self # return the opencv wrapper to context manager if vcam dont work

    def __exit__(self, exc_type, exc_value, traceback):
        if self.vcam:
            self.vcam.close()

    def is_vcam_on(self) -> bool:
        return self.vcam is not None
    
    def show(self, winname: str, mat: np.ndarray) -> None:
        # display window 
        cv.imshow(winname, mat)

        # send to vcam
        if self.vcam is not None:
            image_rgb = cv.cvtColor(mat, cv.COLOR_BGR2RGB)
            self.vcam.send(image_rgb)
            self.vcam.sleep_until_next_frame()