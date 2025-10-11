from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import webrtcvad
    import sounddevice as sd
    _HAS_AUDIO = True
except Exception:
    _HAS_AUDIO = False

@dataclass
class state_VAD:
    available: bool
    activity: bool = False


class VADStream:
    """
    the strategy to turn on/off VAD is using hysteresis
    has to be on for a couple frames and off for a couple frames
    this prevents extreem annoying jittering
    """
    # attack and release are how many frames it takes to release from hysteresis
    def __init__(self, 
                sample_rate: int = 48000,
                frame_ms: int = 30, 
                aggressiveness: int = 2, 
                attack: int = 4, 
                release: int = 10, 
                device: any = None, 
                ):
        
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.aggressiveness = aggressiveness
        self.attack = attack
        self.release = release
        self.device = device

        self.state = state_VAD(available=_HAS_AUDIO, activity = False)

        # internal implementations shouldnt be used outside of this class
        self._vad = webrtcvad.Vad(aggressiveness) if _HAS_AUDIO else None # creates the VAD object
        self._stream = None # audio input
        self._s_count = 0 # the amount of frames speech was counted
        self._q_count = 0 # quite frames count

    def start(self):
        if not _HAS_AUDIO:
            return
        # how many samples per audio chunk
        blocksize = int(self.frame_ms * self.sample_rate / 1000)

        self._stream = sd.InputStream(
            channels = 1,
            dtype = "int16",
            samplerate = self.sample_rate,
            blocksize = blocksize,
            device = self.device,
            callback = self.callback,
        )

        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream = None # clear object after stopping

    # called every audio chunk depending on frame_ms (buffer size)
    def callback(self, indata: np.ndarray, frames: int, time, status):
        if not _HAS_AUDIO:
            return
        # since webrtc needs audio bits convert to bytes()
        buffer = bytes(indata)
        try:
            speech = self._vad.is_speech(buffer, self.sample_rate)
        except Exception:
            speech = False

        if speech:
            self._s_count += 1
            self._q_count = 0
            if not self.state.activity and self._s_count >= self.attack:
                self.state.activity = True
        else:
            self._q_count += 1
            self._s_count = 0
            if self.state.activity and self._q_count >= self.release:
                self.state.activity = False

