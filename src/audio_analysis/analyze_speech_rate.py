# Import required libraries
import json
import sys
import os
import argparse
from collections import defaultdict

# Setup command-line argument parser
parser = argparse.ArgumentParser(description="Analyze speech rate from timestamped words.")
parser.add_argument("timestamp_path", help="Path to timestamps JSON file")
args = parser.parse_args()

# Load word-level timestamp data from a JSON file
def load_word_timestamps(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# Compute speech rate using a sliding window approach
# window_size: duration in seconds of the analysis window
# fast_threshold: WPM above which the segment is considered too fast
# slow_threshold: WPM below which the segment is considered too slow
def compute_speech_rate(words, window_size=5.0, fast_threshold=180, slow_threshold=90):
    results = []
    i = 0
    n = len(words)

    while i < n:
        start_time = words[i]['start_time']
        end_time = start_time + window_size

        # Collect all words that fall within the current time window
        window_words = []
        while i < n and words[i]['start_time'] <= end_time:
            window_words.append(words[i])
            i += 1

        if not window_words:
            continue

        # Calculate speech rate: words per minute
        duration_min = (window_words[-1]['end_time'] - window_words[0]['start_time']) / 60.0
        word_count = len(window_words)
        wpm = word_count / duration_min if duration_min > 0 else 0

        # Determine if speech rate is too fast or too slow
        status = None
        if wpm > fast_threshold:
            status = "fast"
        elif wpm < slow_threshold:
            status = "slow"

        # Save the result if the speech rate is outside normal range
        if status:
            results.append({
                "start_time": round(window_words[0]['start_time'], 2),
                "end_time": round(window_words[-1]['end_time'], 2),
                "wpm": round(wpm, 2),
                "status": status
            })

    return results

# Main function: coordinates loading data, analyzing, and saving output
def main():
    timestamp_json = args.timestamp_path
    if not os.path.isfile(timestamp_json):
        print(f"❌ Error: File '{timestamp_json}' not found.")
        sys.exit(1)

    words = load_word_timestamps(timestamp_json)
    segments = compute_speech_rate(words)

    # Output results to a JSON file
    output_path = "./tests/results/speech_rate_analysis.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(segments, f, indent=2)

    print(f"✅ Speech rate analysis complete. Output written to {output_path}")

# Entry point of the script
if __name__ == "__main__":
    main()