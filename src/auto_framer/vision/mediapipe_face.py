from __future__ import annotations
import numpy as np
from ..types import Bbox
import cv2 as cv

try:
    import mediapipe as mp
    _HAS_MP = True
except Exception:
    _HAS_MP = False

class FaceDetector:
    """
    thin lil wrapper around MediaPipe Facedetection
    returns faces in absolute coordinates as a Bbox 
    """
    def __init__(self, min_confidence: float = 0.6, model: int = 0):
        """
        model=0 is short range around 2m 
        model=1 is for longer distance around 5m
        """
        self.available = _HAS_MP # checks whether it imported successfully or not
        self.detector = None # mediapipe object 
        if _HAS_MP:
            self._mp_fd = mp.solutions.face_detection # keeps code clearer and helps resusability - docs do this too
            self._detector = self._mp_fd.FaceDetection(
                model_selection=model, min_detection_confidence=min_confidence
            )
    
    def detect(self, frame_BGR: np.ndarray) -> list[Bbox]:
        """
        input: a BGR format picture(standard for OpenCV)
        output: list of Bbox in pixel coordinates like so (x,y,w,h)
        """
        if not self.available or self._detector is None: # extra protection in case self.availabe becomes manually True
            return [] 
        h, w = frame_BGR.shape[:2] # get dimensions
        frame_RGB = cv.cvtColor(frame_BGR, cv.COLOR_BGR2RGB)
        # in the documentation they joined the line above and the one below into one, i didnt it was not as clear
        result = self._detector.process(frame_RGB)

        if not result.detections: # checks for detections(faces)
            return []
        
        boxes: list[Bbox] = []
        for detect in result.detections:
            # changing from relative -> absolute coordinates
            face_rect = np.multiply(
                [
                detect.location_data.relative_bounding_box.xmin,
                detect.location_data.relative_bounding_box.ymin,
                detect.location_data.relative_bounding_box.width,
                detect.location_data.relative_bounding_box.height
                ],
                [w,h,w,h] 
            )

