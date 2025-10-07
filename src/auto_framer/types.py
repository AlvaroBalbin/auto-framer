from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class Bbox:
    x: int
    y: int
    w: int
    h: int 
    # using xywh format instead of xyxy (more common in OpenCV)

    def make_tuple(self) -> tuple[int, int, int, int]:
        return(self.x, self.y, self.w, self.h)
    
    def center(self) -> tuple[int, int]:
        center_x = self.x + (self.w/2)
        center_y = self.y + (self.h/2)
        return center_x, center_y
        

@dataclass(slots=True)
class Track:
    track_id: int
    bbox: Bbox # person facial location
    mouth_activity: float = 0.0 # assume no activity at start
    vad_speaking: bool = False # does vad think speaking is going on
    score: float = 0.0 # visual + audio score to make final decision

# a per-frame info class
@dataclass(slots=True)
class FrameInfo:
    frame_index: int
    frame_time: float # good absolute time reference
    frame_fps: float # the fps measured at that frame


