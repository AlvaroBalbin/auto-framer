from __future__ import annotations
from ..types import Track

class SelectSpeaker:
    """
    decides active speaker depending on global VAD + mouth activity
    speaker based VAD would be better if each user has seperate microphones
    """

    def __init__(self, stickiness: float = 0.08, hold_frames: int = 6):
        """stickiness is the value that assigns a value to how long a current speaker has been talking for
        prevents ping pong, and hold_frames even if score of Tracks changes a minimum amount of time 
        needs to pass before the speaker is switched """
        self._stickiness = stickiness
        self._current_id : int | None = None
        self._hold_frames = hold_frames
        self._age : int = 0

    def select(self, tracks: list[Track], vad_activity: bool) -> int | None:
        if not tracks:
            self._current_id = None
            self._age = 0
            return 
        
        for t in tracks:
            # if vad is silent then lower scores
            base = t.mouth_activity * 13
            t.score = base if vad_activity else 0.4 * base # reduce score half everytime

        # add hysteresis to current track
        if self._current_id is not None:
            for t in tracks:
                if t.track_id == self._current_id: # match track to current speaker
                    t.score += self._stickiness
                    break # after we find speaker no need to continue

        best_track = max(tracks, key=lambda t: t.score)

        if self._current_id is None:
            self._current_id = best_track.track_id
            self._age = 0
            return self._current_id # if its the first frame of speaker then just return that
        
        # if another speaker has far higher by a margin, then you can switch
        margin = 0.03
        current = next((t for t in tracks if t.track_id == self._current_id), None)
        # change the speaker when there is no speaker or the new best speaker is better than current one by + margin
        if current is None or (self._current_id != best_track.track_id and best_track.score > (current.score + margin)):
            self._current_id = best_track.track_id
            self._age = 0
        else:
            # if there is no major difference then increase age
            self._age += 1
            if self._age > self._hold_frames:
                # if age is above threshold _hold_frames then change speaker to best_track (may be none)
                self._current_id = best_track.track_id
                self._age = 0

        return self._current_id