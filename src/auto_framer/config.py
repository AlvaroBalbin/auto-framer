from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class Config:
    # video
    camera_index: int = 0 # which camera to use (0 is default)
    input_width: int = 1280 # dimensons for camera capture
    input_height: int = 720
    target_width: int = 1280 # dimensions for output video
    target_height: int = 720
    fps: int = 30 
    mirror: bool = True # flip horizontally - feels more natural
    window_name: str = "Auto-Framer"
    maximum_failures: int = 30 # number of failed frames before closing

    # overlays
    show_fps: bool = True # fps counter
    show_debug_tools: bool = False # extra debug information
    show_vad_overlay: bool = True # show voice activity detection overlay
    show_bbox: bool = False

    # framing
    framing_tightness: float = 0.8 # how much space to keep around face
    focus_on_speaker: bool = False

    # virtual camera
    enable_virtual_camera: bool = False 

    # filters for smoothing
    alpha_for_ema: float = 0.25 # smoothing factor for exponential moving average
    hysteresis_time: float = 0.75 # seconds to wait before switching states
    
    # audio
    sample_rate: int = 16000 # webrtc VAD only works with 8000, 16000, 32000 or 48000 Hz
    vad_frame_duration: int = 30 # webrtc VAD frame accepts only 10, 20 or 30 ms

CONFIG_DEFAULT = Config()
