from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from datetime import datetime
import argparse


# Convert timestamp hh:mm:ss into seconds
def hms_second(hms):
    return (
        int(hms.split(":")[0]) * 3600
        + int(hms.split(":")[1]) * 60
        + int(hms.split(":")[2])
    )


# For logging purposes
def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    # Setup the Argument Parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument(
        "-d", "--dir", help="Directory of video and where to save", required=True
    )
    parser.add_argument(
        "-f", "--file", help="File of video and where to save", required=True
    )
    parser.add_argument(
        "-t",
        "--times",
        nargs="+",
        help="Times in hh:mm:ss format; where to split the video",
        required=True,
    )

    dict_args = dict(parser.parse_args()._get_kwargs())

    video_dir = dict_args["dir"]
    video_file = video_dir + dict_args["file"]

    # Check total duration
    video = VideoFileClip(video_file)
    duration = video.duration
    print(f"{get_time()}\tVideo duration (s):{duration}")

    # Get the times
    times = dict_args["times"]

    times = [x.strip() for x in times]

    # Loop through all the timestamps that you want to split the video
    starttime = 0  # start time in seconds
    for idx, time in enumerate(times):
        # Ensure the timestamp format is correct
        assert time.count(":") == 2
        assert len(time) == 8

        if idx == 0:
            endtime = hms_second(time)
        else:
            starttime = endtime
            endtime = hms_second(time)
        ffmpeg_extract_subclip(
            video_file,
            starttime,
            endtime,
            targetname=video_file.replace(
                "20231020120405-00.00.00.000-00.24.11.784.mp4", ""
            )
            + "-part"
            + str(idx + 1)
            + ".mp4",  # rename the file however you want
        )
        print(f"{get_time()}\t[DONE] Clip from {starttime} until {endtime}")
    # the last piece
    if endtime < duration:
        ffmpeg_extract_subclip(
            video_file,
            endtime,
            duration,
            targetname=video_file.replace(
                "20231020120405-00.00.00.000-00.24.11.784.mp4", ""
            )
            + "-part"
            + str(idx + 2)
            + ".mp4",  # rename the file however you want
        )
        print(f"{get_time()}\t[DONE] Clip from {endtime} until {duration}")
