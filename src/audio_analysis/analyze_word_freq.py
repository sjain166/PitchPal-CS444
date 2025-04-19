import json
import os
import argparse
import re
from collections import defaultdict

# --- Command-line argument parsing ---
parser = argparse.ArgumentParser(description="Analyze overused words from transcription.")
parser.add_argument("timestamp_path", help="Path to timestamps JSON file")
parser.add_argument("transcription_path", help="Path to transcription .txt file")
# parser.add_argument("--timestamp_path", default="../tests/timestamp.json", help="Path to timestamps JSON file")
# parser.add_argument("--transcription_path", default="../tests/transcription.txt", help="Path to transcription .txt file")
parser.add_argument("--report_path", default="./tests/results/word_frequency_report.json", help="Path to save frequency report")
args = parser.parse_args()

# Extract argument values
transcription_path = args.transcription_path
timestamp_path = args.timestamp_path
report_path = args.report_path

# Ensure the output directory exists
os.makedirs(os.path.dirname(report_path), exist_ok=True)

# --- Load word-level timestamp data ---
with open(timestamp_path, "r") as f:
    timestamps = json.load(f)

# --- Parameters for overuse detection ---
TIME_WINDOW = 10  # seconds: how wide the sliding time window is
FREQUENCY_THRESHOLD = 3  # the minimum number of repetitions required to consider a word overused
# Common words requiring a higher threshold
COMMON_WORDS = {"i", "the", "in", "and", "to", "a", "of", "is", "it", "that"}
COMMON_FREQ_THRESHOLD = 5

# --- Build a mapping from each word to a list of its timestamped instances ---
word_occurrences = defaultdict(list)
for entry in timestamps:
    word = entry["word"].lower().strip()
    # Consider only clean alphabetical words
    if re.match(r"^[a-zA-Z']+$", word):
        word_occurrences[word].append(entry)
        
# --- Analyze overused words ---
# List to collect report entries in the new format
overused_report = []

for word, instances in word_occurrences.items():
    # Choose threshold based on word category
    threshold = COMMON_FREQ_THRESHOLD if word in COMMON_WORDS else FREQUENCY_THRESHOLD
    # Skip words that appear less than the threshold for this category
    if len(instances) < threshold:
        continue
    
    # Time-based sliding windows to detect overuse without overlap
    sorted_instances = sorted(instances, key=lambda x: x["start_time"])
    window_start = sorted_instances[0]["start_time"]
    last_time = sorted_instances[-1]["start_time"]
    while window_start <= last_time:
        window_end = window_start + TIME_WINDOW
        window_hits = [inst for inst in sorted_instances if window_start <= inst["start_time"] <= window_end]
        if len(window_hits) >= threshold:
            overused_report.append({
                "word": word,
                "count": len(window_hits),
                "timestamps": [
                    {
                        "start_time": round(inst["start_time"], 2),
                        "end_time":   round(inst["end_time"],   2)
                    }
                    for inst in window_hits
                ]
            })
            # jump ahead by the window size to avoid overlapping detections
            window_start = window_end
        else:
            # slide window by one second when no overuse detected
            window_start += 1

# --- Save analysis to JSON file ---
with open(report_path, "w") as f:
    json.dump(overused_report, f, indent=4)

print(f"✅ Overused Words Report saved to: {report_path}")