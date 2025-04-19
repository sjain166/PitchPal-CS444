import argparse
import subprocess
import os
import glob
import sys

def run_script(script_path, args_list):
    result = subprocess.run(["python3", script_path] + args_list)
    if result.returncode != 0:
        print(f"❌ Error running {script_path}")
        sys.exit(1)
        
def main():
    parser = argparse.ArgumentParser(description="Run CrisperWhisper transcription on an audio file.")
    parser.add_argument("audio_path", help="Path to the input audio file (e.g., ../data/Pitch-Sample/sample01.wav)")

    args = parser.parse_args()
    audio_path = args.audio_path

    # Validate file
    if not os.path.isfile(audio_path):
        print(f"❌ Error: File '{audio_path}' does not exist.")
        sys.exit(1)

    # Export as environment variable (optional) or pass as argument
    print("🚀 Starting transcription...")
    subprocess.run(["python3", "preprocessing.py", audio_path])
    
    # Define paths
    timestamp_json = "./tests/timestamp.json"
    transcription_txt = "./tests/transcription.txt"
    analysis_folder = "./audio_analysis"
    results_folder = "./tests/results"
    os.makedirs(results_folder, exist_ok=True)
    
    # Run 3 analysis scripts once
    print("🧠 Running global analysis scripts...")
    analysis_scripts_once = [
        "analyze_profanity.py",
        "analyze_filler.py",
        "analyze_volume.py",
        "analyze_speech_rate.py",
        "analyze_word_freq.py",
        "analyze_emotion.py",
        "analyze_sentence_structure.py"
    ]
    for script in analysis_scripts_once:
        script_path = os.path.join(analysis_folder, script)
        if script == "analyze_volume.py":
            run_script(script_path, [audio_path])
        elif script == "analyze_speech_rate.py" or script == "analyze_filler.py":
            run_script(script_path, [timestamp_json])
        elif script == "analyze_emotion.py":
            run_script(script_path, [audio_path, timestamp_json])
        else:
            run_script(script_path, [timestamp_json, transcription_txt])

    print("✅ All tasks completed successfully.")

if __name__ == "__main__":
    main()