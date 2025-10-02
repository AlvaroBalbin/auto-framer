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
        new_tracks: dict[int, Track] = []

        for detect in detections:
            # prepare variable for greedy search
            closest_id: int = None
            closest_distance = self.max_distance

            for track_id in list(unmatched_tracks):
                dist = self._distance(detect, self._tracks[track_id])
                if dist < closest_distance:
                    closest_distance = dist
                    closest_id = track_id
                
            if closest_id is not None:
                # map track if it exists
                track = self._tracks[closest_id]
                track.bbox = detect # insert new bbox into previous frames track
                new_tracks[closest_id] = track
            else:
                # make a new track cause new face
                new_id = self._next_id
                self._next_id += 1
                new_tracks[track_id] = Track(track_id=new_id, bbox=detect)
                unmatched_tracks.remove(closest_id)

        # make sure you add age to unmatched ones
        for track_id in unmatched_tracks:
            self._ages[track_id] += 1

        # remove too old tracks
        alive = {track_id: track for track_id, track in new_tracks.items()}
        for track_id, age in self._ages.items():
            if age > self.max_age:
                self._ages.pop(track_id, None)

        # reset tracks for active track
        for track_id in alive.keys():
            self._ages[track_id] = 0

        self._tracks = new_tracks

        return list(self._tracks.values())
    
    




