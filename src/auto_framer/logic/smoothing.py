from __future__ import annotations
from ..types import Bbox

class EMASmoother:
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self._state = tuple[float, float, float, float] | None = None

    # call reset when track face changes, on boot-up or also just no faces detected
    def reset(self) -> None:
        self._state = None

    def update(self, crop: Bbox) -> Bbox:
        # get current detected size
        x, y, w, h = crop.make_tuple

        if self._state is None:
            # theres no smoothed previous frame -> we have to assign values to it
            self._state = (float(x), float(y), float(w), float(h))
        else:
            # apply EMA to each coordinate
            newx, newy , neww, newh = self._state
            self._state = (
                x + self.alpha * (newx - x),
                y + self.alpha * (newy - y),
                w + self.alpha * (neww - w),
                h + self.alpha * (newh - h),
            )
            sx, sy, sw, sh = self._state
            # cannot return it as a float -> cant have half pixels
            return Bbox(int(round(sx)), int(round(sy)), int(round(sw)), int(round(sh)))
