from __future__ import annotations
import math
from ..types import Bbox, Track

class Tracker:
    """
    simple centroid based tracker for the face boxes
    keeps relatively stable IDs as detection stay close between frames
    however it is not Hungarian assigment not global checking, should not be necessary
    """

    def __init__(self, max_distance: float = 80.0, max_age: int = 5):
        self.max_distance = max_distance
        self.max_age = max_age
        self._next_id = 1
        self._tracks: dict[int, Track] = {} # active tracks found by track_id
        self._ages: dict[int, int] = {} # how many frames since each track was last matched

    def _distance(self, a: Bbox, b: Bbox) -> float:
        # a is Bbox from new detection, b is the Bbox from previous frame detection
        ax, ay = a.center()  # call center() function since a and b are Bbox's
        bx, by = b.center()
        distance = math.hypot(ax-bx, ay-by)
        return distance
    
    def update(self, detections: list[Bbox]) -> list[Bbox]:
        # unassign all current tracks
        unmatched_tracks = set(self._tracks.keys())

