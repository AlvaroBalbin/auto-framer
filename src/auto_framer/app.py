from __future__ import annotations
import time
import cv2 as cv
import numpy as np

from .config import CONFIG_DEFAULT
from .video.capture import VideoCapture
from .video.renderer import Renderer
from .utils.timers import FpsCalculator
from .ui import overlays
from .ui.keyboard import KeyPolls
from .types import FrameInfo, Bbox, Track


def crop_center_16x9(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    webcams can have many different sizes/rations to fix this
    we just "zoom" into the picture until we get the right resolution.
    center-crop to out_w and out_h, then just resize to those dimensions.
    """
    in_h, in_w = frame.shape[:2] # slice the output tuple

    # calculate aspect ratios
    target_aspect = out_w / out_h
    current_aspect = in_w / in_h

    if current_aspect > target_aspect:
        # two wide rn -> crop width
        new_w = int(in_h * target_aspect)
        xzero = (in_w - new_w) // 2 # you may get one pixel larger on one side - invisible asymmetry
        crop = frame[:, xzero: xzero + new_w, :] # explicit slicing

    elif current_aspect < target_aspect:
        # two tall -> crop height
        new_h = int(in_w * target_aspect)
        yzero = (in_h - new_h) // 2
        crop = frame[yzero: yzero + new_h, :,  :] # very explicit slicing (: are not neccessary really)
    else:
        # perfect aspect ratio - lucky!
        crop = frame
    return cv.resize(crop, (out_w, out_h), interpolation=cv.INTER_LINEAR)

def main():
    # all parameters for configuration object come from config.py
    cfg = CONFIG_DEFAULT

    # call personalized wrapper - not actual videocapture function from OpenCV
    cap = VideoCapture(cfg.camera_index, (cfg.input_width, cfg.input_height), cfg.fps)
    cap.open()

    # create instance of class(object) for fps information
    fps_counter = FpsCalculator()
    
    # optional but it allow for more specific window settings rather than just cv.imshow()
    cv.namedWindow(cfg.window_name, cv.WINDOW_AUTOSIZE)

    with Renderer(cfg.target_width, cfg.target_height, cfg.fps, cfg.enable_virtual_camera) as rend:
        frame_index = 0
        while True:
            retval, frame = cap.read()
            if not retval or frame is None:
                break

            if cfg.mirror:
                frame = cv.flip(frame, 1)

            # center crop to 16:9 aspect ratio and resize
            crop_out = crop_center_16x9(frame, cfg.input_width, cfg.input_height)

            # overlays 
            fps = fps_counter.tick()
            info = FrameInfo(frame_index=frame_index, frame_time=time.perf_counter, frame_fps=fps)
            if cfg.show_fps:
                overlays.draw_fps(crop_out, info)

            rend.show(cfg.window_name, crop_out) 

            # keyboard polls
            ks = KeyPolls(1) # 1 ms delay per poll
            if ks.quit:
                break
            if ks.toggle_mirror:
                cfg.mirror = not cfg.mirror

            frame_index += 1

        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()






    


