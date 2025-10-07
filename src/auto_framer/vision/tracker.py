from __future__ import annotations
import math
from ..types import Bbox, Track
import heapq

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
        self._free_ids: list[int] = [] # use a heap so you can get lowest value in O(1) when renaming tracks

    def _distance(self, a: Bbox, b: Bbox) -> float:
        # a is Bbox from new detection, b is the Bbox from previous frame detection
        ax, ay = a.center()  # call center() function since a and b are Bbox's
        bx, by = b.center()
        distance = math.hypot(ax-bx, ay-by)
        return distance
    
    def update(self, detections: list[Bbox]) -> list[Track]:
        # code needs these to tell which tracks are matched, created, unmatched
        # unassign all current tracks
        unmatched_tracks = set(self._tracks.keys())
        new_tracks: dict[int, Track] = {}

        matched_ids: set[int] = set()
        new_ids: set[int] = set()

         # make sure you add age to unmatched ones
        for track_id in unmatched_tracks:
            self._ages[track_id] = self._ages.get(track_id, 0) + 1

        # dont iterate over a dict so we put it into a list then delete it in the following loop
        # remove too old tracks
        to_remove = []
        for track_id, age in self._ages.items():
            if age >= self.max_age:
                to_remove.append(track_id)

        for track_id in to_remove:
            self._ages.pop(track_id, None)
            self._tracks.pop(track_id, None)
            heapq.heappush(self._free_ids, track_id)


        # remove all removed tracks from unmatched_tracks using set difference
        unmatched_tracks -= set(to_remove)

        for detect in detections:
            # prepare variable for greedy search
            closest_id: int | None = None
            closest_distance = self.max_distance

            for track_id in list(unmatched_tracks):
                dist = self._distance(detect, self._tracks[track_id].bbox)
                if dist < closest_distance:
                    closest_distance = dist
                    closest_id = track_id
                
            if closest_id is not None:
                # map track if it exists
                track = self._tracks[closest_id]
                matched_ids.add(closest_id)

                track.bbox = detect # insert new bbox into previous frames track
                unmatched_tracks.remove(closest_id) # have to remove it after it gets matched
                new_tracks[closest_id] = track
            else:
                # make a new track cause new face
                if self._free_ids:
                    # get the newest id if possible
                    new_id = heapq.heappop(self._free_ids)
                else:
                    # just get newest id if free_ids is False
                    new_id = self._next_id
                    self._next_id += 1

                new_tracks[new_id] = Track(track_id=new_id, bbox=detect)
                new_ids.add(new_id)
                self._ages[new_id] = 0

        for track_id in unmatched_tracks:
            if track_id in  self._tracks and self._ages.get(track_id, 0) < self.max_age:
                new_tracks[track_id] = self._tracks[track_id]

        # reset tracks for active track
        for track_id in (matched_ids | new_ids):
            self._ages[track_id] = 0

        if to_remove:
            # sort remaining track Ids so we assign new ids in order
            ordered_ids = sorted(new_tracks.keys())

            # now create mapping old_id -> new_id so that we dont get massive id numbers
            # basically we assign the old_id to a new fresh key starting from 1 not 0
            id_map = {old_id: i + 1 for i , old_id in enumerate(ordered_ids)}

            # now rebuild new_tracks dictionary with new compact id ordering
            compacted_tracks = {}
            compacted_ages = {}

            for old_id, track in new_tracks.items():
                new_id = id_map[old_id]
                track.track_id = new_id # update track object itself
                compacted_tracks[new_id] = track
                compacted_ages[new_id] = self._ages.get(old_id, 0)

            # now that we got compacted tracks in the right order we can synthesize this into the right variables
            new_tracks = compacted_tracks
            self._ages = compacted_ages
            self._next_id = len(new_tracks) + 1 # need +1 so no error is raised
            self._free_ids.clear()

        self._tracks = new_tracks

        return list(self._tracks.values())
    