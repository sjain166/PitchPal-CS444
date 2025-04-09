import json
import os
import argparse
import re
from collections import defaultdict

parser = argparse.ArgumentParser(description="Analyze overused words from transcription.")
parser.add_argument("timestamp_path", help="Path to timestamps JSON file")
parser.add_argument("transcription_path", help="Path to transcription .txt file")
parser.add_argument("--report_path", default="./tests/results/word_frequency_report.json", help="Path to save frequency report")
args = parser.parse_args()

transcription_path = args.transcription_path
timestamp_path = args.timestamp_path
report_path = args.report_path
os.makedirs(os.path.dirname(report_path), exist_ok=True)

# Load timestamps JSON
with open(timestamp_path, "r") as f:
    timestamps = json.load(f)

# Parameters for detecting overuse
TIME_WINDOW = 10  # seconds
FREQUENCY_THRESHOLD = 3  # word appears >= this in the time window

# Build a map from word -> list of timestamp entries
word_occurrences = defaultdict(list)
for entry in timestamps:
    word = entry["word"].lower().strip()
    if re.match(r"^[a-zA-Z']+$", word):
        word_occurrences[word].append(entry)
        
# Track overused words with their timestamps
overused_words = {}

for word, instances in word_occurrences.items():
    if len(instances) < FREQUENCY_THRESHOLD:
        continue

    time_stamps = sorted([i["start_time"] for i in instances])
    for i in range(len(time_stamps) - FREQUENCY_THRESHOLD + 1):
        window = time_stamps[i:i + FREQUENCY_THRESHOLD]
        if window[-1] - window[0] <= TIME_WINDOW:
            overused_words[word] = {
                "type": "content",
                "count": len(instances),
                "instances": [{"start_time": inst["start_time"], "end_time": inst["end_time"]} for inst in instances]
            }
            break

# Save to report
with open(report_path, "w") as f:
    json.dump(overused_words, f, indent=4)

print(f"✅ Overused Words Report saved to: {report_path}")