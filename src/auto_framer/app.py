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
from .logic.framing import compute_crop
from .logic.smoothing import EMASmoother
from .vision.tracker import Tracker
from .vision.mediapipe_face import FaceDetector
from .vision.mouth import MouthActivityEstimator
from .audio.vad import VADStream
from .logic.fusion import SelectSpeaker


def crop_center_16x9(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    webcams can have many different sizes/rations to fix this
    we just crop and resize then we get right resolution.
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
        new_h = int(in_w / target_aspect)
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
    cv.namedWindow(cfg.window_name, cv.WINDOW_NORMAL)

    # we get smoothed frames object, initialize objects such as tracker and face detector to use in the loop
    smoother = EMASmoother(alpha=cfg.alpha_for_ema)
    tracker = Tracker()
    fd = FaceDetector()

    # now we set up appropriate objects for VAD, FaceMesh and the fusion setup to determine score
    mouth = MouthActivityEstimator(alpha=0.3, max_num_faces=4)
    # will default to auto device if None is passed (default in __init__ function)
    vad = VADStream(sample_rate = 16000, frame_ms= 30, aggressiveness = 2, attack = 4, release = 10)
    vad.start()
    speaker = SelectSpeaker(stickiness=0.3, hold_frames=30)


    with Renderer(cfg.target_width, cfg.target_height, cfg.fps, cfg.enable_virtual_camera) as rend:
        frame_index = 0 # starting frame
        failed_frame_counter = 0 # counter for how many frames failed
        speaker_id = None # initialize so that its not local later on -> use it in the HUD overlay
        crop_box = None
        while True:
            retval, frame = cap.read()
            if not retval or frame is None:
                failed_frame_counter += 1
                if failed_frame_counter >= cfg.maximum_failures:
                    print("camera connection got lost, oops")
                    break
                continue
            else:
                # reset after a correct frame passes
                failed_frame_counter = 0

            if cfg.mirror:
                frame = cv.flip(frame, 1)

            # ----------------------------------------------------------------------------
            # face detection and setting tracks
            boxes = fd.detect(frame) # returns list[Bbox] -> might be multiple if + faces are found
            tracks = tracker.update(boxes)

            # TEMPORARY DEBUG used it before I created debug was very useful
            # cv.putText(frame, f"focus={cfg.focus_on_speaker} faces={len(tracks)}",
            # (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv.LINE_AA)
            
            # update mouth data
            mouth.update(frame_bgr = frame, tracks = tracks)

            if cfg.focus_on_speaker:
                if tracks:
                    speaker_id = speaker.select(tracks, vad.state.activity)
                    chosen_speaker = next((t for t in tracks if t.track_id == speaker_id), None)

                    target_bbox = chosen_speaker.bbox if chosen_speaker is not None else None

                    if chosen_speaker is None:
                        # have this as a backup in case the chosen_speaker cannot be determined -> picks largest face
                        chosen_speaker = max((t.bbox for t in tracks), key=lambda b: b.w * b.h) if tracks else None
                    
                    crop_box = compute_crop(target=target_bbox, 
                                            frame_w=frame.shape[1], 
                                            frame_h=frame.shape[0], 
                                            tightness=cfg.framing_tightness)
                    
                    # after computing the crop then smooth it out so we it does not jitter
                    crop_box = smoother.update(crop_box)

                    # to prevent empty arrays from frame -> when bbox goes out of frame and EMA pushes box outside by few pixels
                    H, W = frame.shape[:2] # CURRENT width and height
                    x, y, w, h = crop_box.make_tuple()

                    # convert to int to ensure right data type
                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)

                    # now clamp to image bounds
                    x = max(0, min(x, W-1))
                    y = max(0, min(y, H-1))
                    w = max(1,min(w, W-x))
                    h = max(1, min(h, H-y))

                    # slicing and then resizing
                    cropped = frame[y:y+h, x:x+w]
                    crop_out = cv.resize(cropped, (cfg.target_width, cfg.target_height), interpolation=cv.INTER_LINEAR)

                else:
                    # reset smoothing since it has internal memory
                    smoother.reset()
                    # if the user wants to focus on speaker but no faces are found we will get an error without this fallback
                    crop_out = crop_center_16x9(frame, cfg.target_width, cfg.target_height) 

            else:
                # center crop to 16:9 aspect ratio and resize: if speaker focus on speaker is not True
                crop_out = crop_center_16x9(frame, cfg.target_width, cfg.target_height)


            # overlays
            overlay_frame = crop_out.copy()
            fps = fps_counter.tick() # get what by using the output frame
            info = FrameInfo(frame_index=frame_index, frame_time=time.perf_counter(), frame_fps=fps)
            if cfg.show_fps:
                overlays.draw_fps(overlay_frame, info)

            # show bounding boxes -> what the program considers a face
            if cfg.show_bbox and not cfg.focus_on_speaker:
                overlays.draw_faces(overlay_frame, tracks)   # draw bbox on source frame

             # display the debug hud -> gives extra info on variables like VAD activity
            if cfg.show_debug_tools:
                overlays.draw_hud(frame=overlay_frame,
                                  info=info,
                                  tracks=tracks,
                                  active_id=speaker_id if cfg.focus_on_speaker else None,
                                  vad_speaking=vad.state.activity,
                                  tightness=cfg.framing_tightness,
                                  ema_alpha=cfg.alpha_for_ema,
                                  vcam_on=cfg.enable_virtual_camera,
                                  hud_top_left = cfg.hud_top_left)


            
            rend.show(cfg.window_name, overlay_frame) 

            # keyboard polls
            ks = KeyPolls(1) # 1 ms delay per poll
            if ks.quit: break
            if ks.toggle_mirror: cfg.mirror = not cfg.mirror
            if ks.bbox_overlay: cfg.show_bbox = not cfg.show_bbox
            if ks.focus_on_speaker: cfg.focus_on_speaker = not cfg.focus_on_speaker
            if ks.tighter: cfg.framing_tightness = min(0.98, cfg.framing_tightness + 0.05) # clamp so zooming isnt excessive
            if ks.looser: cfg.framing_tightness = max(0.02, cfg.framing_tightness - 0.05) 
            if ks.toggle_debug_tools: cfg.show_debug_tools = not cfg.show_debug_tools

            frame_index += 1

        vad.stop()
        # mouth.close() not necessary anymore
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()






    


