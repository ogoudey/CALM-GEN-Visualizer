import rerun as rr
import pandas as pd
import numpy as np
from datetime import datetime
import cv2

# print statements are unneeded. run `python3 rerun_viz.py` in a terminal, and open rerun.io
# keep in mind the seconds_to_slide_video argument.

# Please add to:
seconds_to_slide_video_registry = {
    "data/010 1/010/010_Standard.mp4": 12.77,
    "data/010 1/010/010_AI.mp4": 19.17,
}

def display_for_analysis(calibration_path_str: str, data_path_str: str, video_path_str: str, seconds_to_slide_video: float):
    """
        seconds_to_slide_video: Go to T0 of data in the video. Get the seconds already played in the video.
    """
    from process import analyze_world_interaction
    (
        timestamps,
        pupil_sizes,
        brightnesses,
        predicted_pupil_sizes,
        difference,
        arousal,
    ) = analyze_world_interaction(calibration_path_str, data_path_str)

    rr.init("CALM-GEN Analysis", spawn=True)
    rr.spawn()
    timestamps_dt = pd.to_datetime(timestamps)
    t0 = timestamps_dt[0]
    print(f"T0: {t0}")
    timestamps_s = (timestamps_dt - t0).total_seconds().to_numpy()
    print(f"timestamps_s: len {len(timestamps_s)}: {timestamps_s}")
    print(f"timestamps_s: len {len(timestamps_s)}: {timestamps_s}")
    print(f"differences: {difference}")
    for i, t_s in enumerate(timestamps_s):
        rr.set_time("time", timestamp=t_s)

        rr.log("analysis/pupil/observed_mm", rr.Scalars(pupil_sizes[i]))
        rr.log("analysis/pupil/predicted_mm", rr.Scalars(predicted_pupil_sizes[i]))
        rr.log("analysis/brightness", rr.Scalars(brightnesses[i]))
        rr.log("analysis/difference", rr.Scalars(difference[i]))
        if i < len(arousal):
            rr.log("analysis/arousal", rr.Scalars(arousal[i]))


    cap = cv2.VideoCapture(video_path_str)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    video_start_s = timestamps_s[0] - seconds_to_slide_video
    frame_times_s = video_start_s + np.arange(frame_count) / fps

    frame_times_video_ns = (np.arange(frame_count) / fps * 1e9).astype(np.int64)

    video = rr.AssetVideo(path=video_path_str)
    rr.log("video", video, static=True)
    print("Timeline start:", frame_times_s[0])
    print("Timeline end:", frame_times_s[-1])
    rr.send_columns(
        "video",
        indexes=[rr.TimeColumn("time", timestamp=frame_times_s)],
        columns=rr.VideoFrameReference.columns_nanos(frame_times_video_ns),
    )

import sys
if __name__ == "__main__":
    calibration_path_str = sys.argv[1]
    data_path_str = sys.argv[2]
    video_path_str = sys.argv[3]
    if len(sys.argv) > 4:
        seconds_to_slide_video = float(sys.argv[4])
    else:
        try:
            seconds_to_slide_video = seconds_to_slide_video_registry[video_path_str]
        except KeyError:
            print("This video does not have a value for how much to slide it in the internal registry. Please add to the registry or provide a number (e.g. 0) at `python3 pc pd v <number>`")
            sys.exit(1)
    display_for_analysis(calibration_path_str, data_path_str, video_path_str, seconds_to_slide_video)