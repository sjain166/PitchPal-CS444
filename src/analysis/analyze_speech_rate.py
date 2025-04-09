import json
import sys
import os
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description="Analyze speech rate from timestamped words.")
parser.add_argument("timestamp_path", help="Path to timestamps JSON file")
args = parser.parse_args()

def load_word_timestamps(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def compute_speech_rate(words, window_size=5.0, fast_threshold=180, slow_threshold=90):
    """
    Sliding window of window_size seconds to compute WPM
    """
    results = []
    i = 0
    n = len(words)

    while i < n:
        start_time = words[i]['start_time']
        end_time = start_time + window_size

        # collect words in this window
        window_words = []
        while i < n and words[i]['start_time'] <= end_time:
            window_words.append(words[i])
            i += 1

        if not window_words:
            continue

        duration_min = (window_words[-1]['end_time'] - window_words[0]['start_time']) / 60.0
        word_count = len(window_words)
        wpm = word_count / duration_min if duration_min > 0 else 0

        status = None
        if wpm > fast_threshold:
            status = "fast"
        elif wpm < slow_threshold:
            status = "slow"

        if status:
            results.append({
                "start_time": round(window_words[0]['start_time'], 2),
                "end_time": round(window_words[-1]['end_time'], 2),
                "wpm": round(wpm, 2),
                "status": status
            })

    return results

def main():
    timestamp_json = args.timestamp_path
    if not os.path.isfile(timestamp_json):
        print(f"❌ Error: File '{timestamp_json}' not found.")
        sys.exit(1)

    words = load_word_timestamps(timestamp_json)
    segments = compute_speech_rate(words)

    output_path = "./tests/results/speech_rate_analysis.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(segments, f, indent=2)

    print(f"✅ Speech rate analysis complete. Output written to {output_path}")

if __name__ == "__main__":
    main()