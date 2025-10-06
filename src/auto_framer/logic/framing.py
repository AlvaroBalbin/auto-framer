from __future__ import annotations
from ..types import Bbox

def _16x9to(w: int, h: int) -> tuple[int, int]:
    aspect = 16 / 9
    current_aspect = w / h

    if current_aspect > aspect:
        # too wide -> increase height
        new_h = int(round(w / aspect))
        return w, new_h
    else:
        # too tall -> increase width
        new_w = int(round(h * aspect))
        return new_w, h
    
def compute_crop(target: Bbox | None, frame_w: int, frame_h: int, tightness: float) -> Bbox:
    """
    returns a Bbox of dimensions (x,y,w,h), which acts as a crop for our specific target
    tightness is the following 1 = very zoomed in, 0.5 = looser framing
    and if there is no target displays full frame
    """

    if target is None:
        w = min(frame_w, int(frame_h * 16 / 9))
        h = min(frame_h, int(frame_w * 9 / 16))
        x = (frame_w - w) // 2
        y = (frame_h - h) // 2
        return Bbox(x, y, w, h)
    
    # set up our desired width and height following intensity of tightness
    targw = max(1, int(target.w * (1 / tightness)))
    targh = max(1, int(target.h * (1 / tightness)))

    # then force the shape into correct aspect ratio
    truew, trueh = _16x9to(targw, targh)

    # clamp to frame ensuring it fits
    truew = min(truew, frame_w)
    trueh = min(trueh, frame_h)

    cx, cy = target.center()
    # the frame might be too close to corners so we need to ensure that start position isnt out of bounds
    x = max(0, min(frame_w - truew, cx - truew // 2))
    y = max(0, min(frame_h - trueh, cy - trueh // 2))

    return Bbox(x, y, truew, trueh)