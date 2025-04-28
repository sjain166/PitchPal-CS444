#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

def extract_audio(video_path: Path, audio_path: Path):
    cmd = [
        "ffmpeg",
        "-y",                # overwrite if exists
        "-i", str(video_path),
        "-vn",               # no video
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        str(audio_path)
    ]
    print(f"🎬 Extracting audio: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"✅ Audio extracted to {audio_path}")

def run_script(script: Path, args: list[str]):
    cmd = ["python3", str(script)] + args
    print(f"🚀 Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(
        description="Unified runner: video→audio ML pipelines."
    )
    parser.add_argument(
        "--video", "-v", required=True,
        help="Path to input video file"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"❌ Error: video '{video_path}' not found.")
        sys.exit(1)

    # 1) extract audio
    audio_path = Path("output/extracted_audio.wav")
    try:
        extract_audio(video_path, audio_path)
    except subprocess.CalledProcessError:
        print("❌ ffmpeg failed; aborting.")
        sys.exit(1)

    # 2) run video automation
    # try:
    #     run_script(Path("video_automation.py"), ["--video", str(video_path)])
    # except subprocess.CalledProcessError:
    #     print("❌ Video analysis failed; aborting.")
    #     sys.exit(1)

    # 3) run audio automation (this will also launch React at the end)
    try:
        run_script(Path("audio_automation.py"), [str(audio_path)])
    except subprocess.CalledProcessError:
        print("❌ Audio analysis failed; aborting.")
        sys.exit(1)

    print("🎉 All done! If React didn’t start, rerun `npm start` in ./app.")

if __name__ == "__main__":
    main()