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
    subprocess.run(["python3", "transcribe_audio.py", audio_path])
    
    # Define paths
    timestamp_json = "./tests/timestamp.json"
    transcription_txt = "./tests/transcription.txt"
    analysis_folder = "./analysis"
    results_folder = "./tests/results"
    os.makedirs(results_folder, exist_ok=True)
    
    # Run 3 analysis scripts once
    print("🧠 Running global analysis scripts...")
    analysis_scripts_once = [
        # "analyze_profanity.py",
        # "analyze_sentence_structure.py",
        # "analyze_word_freq.py",
    ]
    for script in analysis_scripts_once:
        script_path = os.path.join(analysis_folder, script)
        run_script(script_path, [timestamp_json, transcription_txt])
    
    # Run 4th script per chunk
    print("🔁 Running per-chunk analysis...")
    chunk_files = sorted(glob.glob("./tests/temp_chunk_*.wav"))

    confidence_script = os.path.join(analysis_folder, "analyze_confidence.py")

    for chunk_path in chunk_files:
        chunk_name = os.path.splitext(os.path.basename(chunk_path))[0]
        chunk_result_dir = os.path.join(results_folder, chunk_name)
        os.makedirs(chunk_result_dir, exist_ok=True)

        # Run analyze_confidence.py (RMS analysis with plot and JSON output)
        json_output_path = os.path.join(chunk_result_dir, "confidence_report.json")
        plot_output_path = os.path.join(chunk_result_dir, "confidence_plot.png")
        run_script(confidence_script, [chunk_path, json_output_path, plot_output_path])

    print("✅ All tasks completed successfully.")

if __name__ == "__main__":
    main()