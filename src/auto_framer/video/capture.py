from __future__  import annotations
import sys

import cv2 as cv # documentation prefers cv over cv2

def _default_camera_backend() -> int:
    '''Select default backend for the os API'''
    if sys.platform.startswith('win32'): 
        return cv.CAP_DSHOW # windows, old but stable
    elif sys.platform.startswith('linux'): 
        return cv.CAP_V4L2 # linux most common
    elif sys.platform.startswith('darwin'): 
        return cv.CAP_AVFOUNDATION # macOS
    else:
        return cv.CAP_ANY # let OpenCV decide - fall back
    
class VideoCapture: 
    """
    wrapper around cv.VideoCapture() to improve its interface 
    and create simpler methods
    """
    def __init__(self, index: int = 0, size: tuple[int, int] = (1280, 720), fps: int=30):
        self.index = index
        self.res_w, self.res_h = size
        self.fps = fps

        self.cap: cv.VideoCapture | None = None

    def open(self) -> None:
        """actually opens webcam, checks backend, applys requested parameters to camera"""
        if self.cap is not None: # dont open twice
            return
        
        # get backend
        backend = _default_camera_backend()

        # opens backend with passed index and backend
        self.cap = cv.VideoCapture(self.index, backend)

        if not self.cap.isOpened():
            self.cap.release()
            # fall back onto CAP_ANY if previous backend did not work
            self.cap = cv.VideoCapture(self.index, cv.CAP_ANY)
        
        # still fails then called SysError
        if not self.cap.isOpened():
            raise SystemError(f"could not open camera on index:{self.index}")
        
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, float(self.res_w))
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, float(self.res_h))
        self.cap.set(cv.CAP_PROP_FPS, float(self.fps))

    def read(self):
        if self.cap is None:
            raise RuntimeError("videocapture not opened. call open() first")
        retval, image = self.cap.read()
        if not retval or image is None:
            # error with camera connection
            return False, None
        
        return True, image
    
    def actual_size(self) -> tuple[int,int]:
        if self.cap is None:
            return (0,0)
        w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
        return (w,h)
    
    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        return None

        



        

