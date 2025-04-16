import json
import os
import argparse
import re
from collections import defaultdict

# --- Command-line argument parsing ---
parser = argparse.ArgumentParser(description="Analyze overused words from transcription.")
parser.add_argument("timestamp_path", help="Path to timestamps JSON file")
parser.add_argument("transcription_path", help="Path to transcription .txt file")
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

# --- Build a mapping from each word to a list of its timestamped instances ---
word_occurrences = defaultdict(list)
for entry in timestamps:
    word = entry["word"].lower().strip()
    # Consider only clean alphabetical words
    if re.match(r"^[a-zA-Z']+$", word):
        word_occurrences[word].append(entry)
        
# --- Analyze overused words ---
overused_words = {}

for word, instances in word_occurrences.items():
    # Skip words that appear less than the threshold
    if len(instances) < FREQUENCY_THRESHOLD:
        continue
    
    # Get the sorted list of start times for all occurrences
    time_stamps = sorted([i["start_time"] for i in instances])
    
    # Apply sliding window over the timestamps to detect bursts
    for i in range(len(time_stamps) - FREQUENCY_THRESHOLD + 1):
        window = time_stamps[i:i + FREQUENCY_THRESHOLD]
        if window[-1] - window[0] <= TIME_WINDOW:
            # If the time between first and last in window is small enough,
            # the word is considered overused
            overused_words[word] = {
                "type": "content",
                "count": len(instances),
                "instances": [{"start_time": inst["start_time"], "end_time": inst["end_time"]} for inst in instances]
            }
            break # Only need one valid window to flag the word

# --- Save analysis to JSON file ---
with open(report_path, "w") as f:
    json.dump(overused_words, f, indent=4)

print(f"✅ Overused Words Report saved to: {report_path}")