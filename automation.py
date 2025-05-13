#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

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

def give_actionable_feedback(combined_path: Path):
    with combined_path.open("r", encoding="utf-8") as f:
        combined = json.load(f)

    prompt = f"""
                You are a professional communication coach with 10+ years of experience specializing in crafting and refining elevator pitches.

                Given a JSON object containing:
                - audio_analysis: includes filler_words, profanity_words, word_frequency, volume, sentence_structure
                - video_analysis: includes eye_discontact, facial_expression_nervousness, background_noise
                - original_transcribed_text: the raw speech transcript

                Your tasks are:
                1. Understand the speaker's intent and context by reading the original_transcribed_text.
                2. Analyze the identified issues from both audio and video analyses to assess communication flaws.
                3. Generate a clear and structured report containing the following sections:
                - Strengths
                - Areas for Improvement
                - Actionable Feedback
                - Polished Version of the Speech (if needed)
                4. Do not output tables, as they are not parseable on the frontend.

                JSON INPUT:
                {json.dumps(combined, indent=2)}
                """

    print("🧠 Generating feedback with LLM...")

    client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",  # or any other available model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    # Append feedback to combined JSON
    combined["feedback"] = response.choices[0].message.content

    # Overwrite the combined file
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print("✅ Feedback added to combined.json")



def combine_json_outputs(output_dir: Path, combined_path: Path):
    combined = {}

    # Load all JSON files
    for file in output_dir.glob("*.json"):
        key = file.stem  # filename without .json
        try:
            with file.open("r", encoding="utf-8") as f:
                combined[key] = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Skipping malformed JSON: {file}")

    # Include transcription.txt if it exists
    transcription_file = output_dir / "transcription.txt"
    if transcription_file.exists():
        with transcription_file.open("r", encoding="utf-8") as f:
            combined["raw_transcribed_text"] = f.read().strip()

    # Write combined output
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"📦 Combined JSON written to {combined_path}")

def main():
    # parser = argparse.ArgumentParser(
    #     description="Unified runner: video→audio ML pipelines."
    # )
    # parser.add_argument(
    #     "--video", "-v", required=True,
    #     help="Path to input video file"
    # )
    # args = parser.parse_args()

    # video_path = Path(args.video)
    # if not video_path.is_file():
    #     print(f"❌ Error: video '{video_path}' not found.")
    #     sys.exit(1)

    # 1) extract audio
    # audio_path = Path("output/extracted_audio.wav")
    # try:
    #     extract_audio(video_path, audio_path)
    # except subprocess.CalledProcessError:
    #     print("❌ ffmpeg failed; aborting.")
    #     sys.exit(1)

    # # 2) run video automation
    # try:
    #     run_script(Path("video_automation.py"), ["--video", str(video_path)])
    # except subprocess.CalledProcessError:
    #     print("❌ Video analysis failed; aborting.")
    #     sys.exit(1)

    # # 3) run audio automation (this will also launch React at the end)
    # try:
    #     run_script(Path("audio_automation.py"), [str(audio_path)])
    # except subprocess.CalledProcessError:
    #     print("❌ Audio analysis failed; aborting.")
    #     sys.exit(1)

    output_dir = Path("/Users/aryangupta/Desktop/UIUC/CURRENT/CS-444/PitchPal-CS444/src/tests/Submission_Results")
    combined_json_path = output_dir / "combined.json"
    combine_json_outputs(output_dir, combined_json_path)
    give_actionable_feedback(combined_json_path)

    print("🎉 All done! If React didn’t start, rerun `npm start` in ./app.")

if __name__ == "__main__":
    main()