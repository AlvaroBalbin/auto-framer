from __future__ import annotations
import math
import numpy as np
import cv2 as cv

from ..types import Bbox, Track

mp_face_mesh = mp.solutions.face_mesh

try:
    import mediapipe as mp
    _HAS_MP = True
except Exception:
    _HAS_MP = False

def _iou(a: Bbox, b: Bbox) -> float:
    """
    get the total intersection over both Bbox
    and also their union and find difference:
    it ensures that the mouth activity is linked to
    the correct face
    """
    # get the right and bottom coordinates for x and y
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h

    # calculate where they overlap x1,y1,x2,y2
    overlap_x1, overlap_y1 = max(a.x, b.x), max(a.y, b.y)
    overlap_x2, overlap_y2 = min(ax2, bx2), min(ay2, by2)

    # calculate height and widht of interesection, variables are quite wordy but its clearer imo
    # max prevents errors arising from no overlap at all
    intersection_w = max(0, overlap_x1 - overlap_x2)
    intersection_h = max(0, overlap_y1 - overlap_y2)

    intersection_area = intersection_w * intersection_h

    if intersection_area == 0:
        return 0.0
    
    # calculates total union area
    union_area = a.w * a.h + b.w * b.h

    # intersection over union area, and max() safeguards against division by zero
    iou = int(intersection_area / max(1, union_area))

    return iou

# helps us get distance between points on the cartesian plane
def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    distance = math.hypot(a[0] - b[0], a[1] - b[1])
    return distance

class MouthActivityEstimator:
    """
    use mediapipe facemesh to estimate per-track(face) mouth activity
    the formula for mouth activity:
    mouth_activity = EMA of (MAR - MAR_EMA_Previous)
    we re not measuring absolute MAR instead change of MAR
    smooth the difference since it can be quite jitter on landmarks
    """

    def __init__(self, alpha: float = 0.3, max_num_faces: int = 4):
        self.available: bool = _HAS_MP
        self._alpha = alpha
        self._mar_state: dict[int, float] = {} # mar state of track_id
        self._activity: dict[int, float] = {} # activity ema of track_id

        if self.available:

            self._fm = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=max_num_faces,
                refine_landmarks=False, # helps give far more detailed landmarks, not necessary here
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self._fm = None

        # facial geometry landmark indices, look at github documentation I linked how you can check it yourself
        # mouth corners (atleast the ones I picked). left: 61, right: 291, upper lip: 0, lower lip(D): 17
        self._L = 61
        self._R = 291
        self._U = 0
        self._D = 17   

        def compute_mar(self, mouth_pixels: list[tuple[int, int]]) -> float:
            L = mouth_pixels[self._L]
            R = mouth_pixels[self._R]
            U = mouth_pixels[self._U]
            D = mouth_pixels[self._D]

            # horizontal distance over vertical distance
            mar = float((R - L) / (D - U))

            return mar
        
        # to calculate iou we need to get the raw coordinates to bbox instead of 468 landmarks
        def landmarks_to_bbox(self,  mouth_pixels: list[tuple[int, int]]) -> Bbox:
            all_x = (x[0] for x in mouth_pixels)
            all_y = (y[1] for y in mouth_pixels)

            left_x, top_y = int(min(all_x)), int(min(all_y))
            right_x, bottom_y = int(max(all_x)), int(max(all_x))

            w = max(1, right_x - left_x)
            h = max(1, bottom_y - top_y)

            return Bbox(left_x, top_y, w, h)
        
        def update(self, frame_bgr: np.ndarray, tracks: list[Track]) -> Bbox:









