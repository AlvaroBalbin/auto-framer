from __future__ import annotations
import cv2

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
    def __init__(self)