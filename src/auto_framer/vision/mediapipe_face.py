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
        self._detector = None # mediapipe object 
        if _HAS_MP:
            self._mp_fd = mp.solutions.face_detection # keeps code clearer and helps resusability - docs do this too
            self._detector = self._mp_fd.FaceDetection(
                model_selection=model, min_detection_confidence=min_confidence
            )
    
    def detect(self, frame_bgr: np.ndarray) -> list[Bbox]:
        """
        input: a BGR format picture(standard for OpenCV)
        output: list of Bbox in pixel coordinates like so (x,y,w,h)
        """
        if not self.available or self._detector is None: # extra protection in case self.available becomes manually True
            return [] 
        frame_height, frame_width = frame_bgr.shape[:2] # get dimensions
        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        # in the documentation they joined the line above and the one below into one, i didnt it was not as clear
        result = self._detector.process(frame_rgb)

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
                [frame_width,frame_height,frame_width,frame_height] 
            )

            x, y, w, h = face_rect
            x = max(0, min(x,frame_width-1)) # either zero or the second last pixel before going off the image
            y = max(0, min(y,frame_height-1))

            
            w = min(w, frame_width - x) # either w or whatever is left of the box(prevents overextending)
            h = min(h, frame_height - y)

            # skip degenerated boxes
            if h<=0.0  or w<=0.0:
                continue

            # cast to integers, also round to closest integers. int() truncates towards zero we dont want that
            xn = int(round(x))
            yn = int(round(y))
            wn = int(round(w))
            hn = int(round(h))

            # if a box rounded downwards from like 0.3 -> 0 we might get an error
            if hn<=0.0  or wn<=0.0:
                continue

            # wrap results in a Bbox and place that object in a list []
            boxes.append(Bbox(x=xn,y=yn,w=wn,h=hn))
        return boxes
    
    def close(self):    
        # currently is useless but will be used for future cleanup :)
        pass



