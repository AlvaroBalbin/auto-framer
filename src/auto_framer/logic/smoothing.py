from __future__ import annotations
from ..types import Bbox

class EMASmoother:
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self._state: tuple[float, float, float, float] | None = None
        self._within_deadband = False

    # call reset when track face changes, on boot-up or also just no faces detected
    def reset(self) -> None:
        self._state = None

    def _deadband(self, 
                  sx, sy, sw, sh, # current smoothed state
                  x, y, w, h, # new bbox
                  px: int = 5, # pixel tolerance for deadband
                  frac = 0.05 # percentage tolerance helps with relative sizing
                  ) -> bool: # whether to consider this a "noise" or not
        
        # return True if new Bbox is close enough to the old one
        noise_or_not = (abs(x - sx) < px and 
        abs(y - sy) < px and 
        abs(w - sw) < max(px, frac * sw) and 
        abs(h - sh) < max(px, frac * sh))

        return noise_or_not

    def update(self, crop: Bbox) -> Bbox:
        # get current detected size
        x, y, w, h = crop.make_tuple()
        

        if self._state is None:
            # theres no smoothed previous frame -> we have to assign values to it
            self._state = (float(x), float(y), float(w), float(h))
            return crop # return first frame immediately so first frame isnt blank
        
        sx, sy, sw, sh = self._state # smoothed variables (previous)
        
        if self._deadband(sx, sy, sw, sh, x, y, w, h): # dont need to add other parameters they are defaults
            # helps ignore noise so that the jitter is reduced unless movement threshold is crossed
            return Bbox(int(round(sx)), int(round(sy)), int(round(sw)), int(round(sh)))


        # apply EMA to each coordinate
        self._state = (
            sx + self.alpha * (x - sx),
            sy + self.alpha * (y - sy),
            sw + self.alpha * (w - sw),
            sh + self.alpha * (h - sh),
            )
        sx, sy, sw, sh = self._state
        # cannot return it as a float -> cant have half pixels
        return Bbox(int(round(sx)), int(round(sy)), int(round(sw)), int(round(sh)))
