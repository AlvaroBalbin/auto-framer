from __future__ import annotations
import cv2 as cv
from ..types import Bbox, Track, FrameInfo
import numpy as np

def draw_fps(frame: np.ndarray, info: FrameInfo):
    """draw the fps on the top right corner"""
    if info.frame_fps > 0:
        cv.putText(
            frame, # image
            f"FPS: {info.frame_fps:.2f}", # text
            (10,30), # positioning from top-left
            cv.FONT_HERSHEY_SIMPLEX, # font
            0.6, # font scale
            (0,255,0), # color
            1, # thickness(1 is the default)
            cv.LINE_AA, # line type (anti-aliasing, makes it more aesthetic)
            )
    
def draw_faces(frame: np.ndarray, tracks: list[Track]):
    """get bbox coordinates, draw rectangle around the face, write track_id next to it"""
    for t in tracks:
        x, y, w, h = t.bbox.make_tuple()
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2, cv.LINE_AA) # make rectangle on the bbox
        cv.putText(
            frame, 
            f"Track ID: {t.track_id}",
            (x, max(0,y-8)), # 8 pixels above bbox but safety so it dont leave the image
            cv.FONT_HERSHEY_SIMPLEX,
            0.5, # dont want the text too large
            (255,0,0), # strong blue
            1,
            cv.LINE_AA, # smooth edges
        )

def draw_vad(frame: np.ndarray, speaking: bool, show: bool):
    """display the vad overlay, dipslaying whether audio is detected or not"""
    if not show:
        return
    text = "VAD: speaking" if speaking else "VAD: not speaking"
    color = (0,255,0) if speaking else (200, 200, 200) # green when speaking else gray
    cv.putText(
        frame,
        text,
        (10,60),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv.LINE_AA,
    )

def draw_crop_box(frame: np.ndarray, crop: Bbox):
    """shows current cropping region: box size of what you wanna focus on"""
    x, y, w, h = crop.make_tuple() # crop is a passed Bbox object
    cv.rectangle(frame, (x,y), (x+w,y+h), (100,100,100), 1, cv.LINE_AA)

def draw_hud(
        frame: np.ndarray,
        info: FrameInfo,
        tracks: list[Track],
        active_id: int | None, # relates to current speaker track in use
        vad_speaking: bool,
        tightness: float,
        ema_alpha: float,
        vcam_on: bool,
        hud_top_left: tuple[int, int] = (10,60), # starting position of the hud
        vad_running: bool = None,  # check if vad stream has started (major debug)
        vad_s_count: int = None,     
        vad_q_count: int = None,   
        mouth_activity: int = None,
):
    x, y = hud_top_left
    color = (255, 0, 0) # blue(BGR)
    height_line = 20 # will later seperate each line 

    vcam_str = "ON" if vcam_on else "OFF"
    vad_running = "-" if None else ("speaking" if vad_running else "stopped")
    no_tracks_str = len(tracks) if tracks else "-"
    active_id_str = active_id if active_id else "-"
    
    # get current score for only active tracks
    if active_id and tracks:
        active_track = next((t for t in tracks if active_id == t.track_id), None)
        if active_track is not None:
            score = getattr(active_track, "score", None)
            score_str = f"{score:.2f}" if score is not None else "-"
        else:
            score_str = "-"
    else:
        score_str = "-"

    lines = [
        f"fps: {info.frame_fps:.1f}",
        f"no. of tracks {no_tracks_str}",
        f"current active id {active_id_str}",
        f"vad speaking? = {vad_speaking}",
        f"is vad stream on? {vad_running}",
        f"tightness: {tightness:.2f}",
        f"ema alpha: {ema_alpha:.2f}",
        f"is vcam on? {vcam_str}",
        f"vad s count: {vad_s_count}",
        f"vad q count:  {vad_q_count}",
        f"score: {score_str}",
        f"mouth activity: {mouth_activity}",
    ]

    for i, text in enumerate(lines):
        cv.putText(frame, text, (x, y + (i * height_line)), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv.LINE_AA,)

    # draw a confirmation dot -> letting user know quickly whether VAD is on or not
    vad_color = (0, 255, 0) if vad_speaking else (0, 0, 255)
    cv.circle(frame, (x + 225, 114), 5, vad_color, 1) # green color 