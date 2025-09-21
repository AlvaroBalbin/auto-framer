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
        

