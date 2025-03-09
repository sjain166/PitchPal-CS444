import os
# Used for reading and processing video frames.

"""
This script extracts frames from a video file and saves them as images in a directory.
It also extracts the audio from the video and saves it as a .wav file.
"""

import cv2
from moviepy import VideoFileClip

video_path = "../data/Pitch-Sample/sample01.mp4"
audio_path = "../data/Pitch-Sample/sample01.wav"
frames_dir = "../data/frames/Pitch-Sample/sample01_frames"

os.makedirs(frames_dir, exist_ok=True)

if not os.path.exists(video_path):
    print(f"❌ Error: Video file not found at {video_path}")
    exit()

clip = VideoFileClip(video_path)
if clip.audio is not None and clip is not None:
    clip.audio.write_audiofile(audio_path)
    print(f"Audio extracted and saved to {audio_path}")
else:
    print("⚠️ No audio found in the video. Skipping audio extraction.")

cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_count += 1
    print(f"Processing frame {frame_count}")


cap.release()
print(f"Extracted {frame_count} frames and saved audio to {audio_path}")

