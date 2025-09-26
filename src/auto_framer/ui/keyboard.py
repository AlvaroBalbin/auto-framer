from __future__ import annotations
import cv2 as cv
from dataclasses import dataclass

@dataclass
class KeyStates:
    quit: bool = False
    toggle_mirror: bool = False
    toggle_vad_overlay: bool = False
    toggle_debug_tools: bool = False
    toggle_vcam: bool = False
    tighter: bool = False
    looser: bool = False

def KeyPolls(delay_ms: int = 1):
    key = cv.waitKey(delay_ms) & 0xFF # keep only ASCII bits
    state = KeyStates()
    if key == ord("q"):
        state.quit = True
    elif key == ord("m"):
        state.toggle_mirror = True
    elif key == ord("v"):
        state.toggle_vad_overlay = True
    elif key == ord("d"):
        state.toggle_debug_tools = True
    elif key == ord("c"):
        state.toggle_vcam = True
    elif key == ord("t"):
        state.tighter = True
    elif key == ord("l"):
        state.looser = True
    return state
