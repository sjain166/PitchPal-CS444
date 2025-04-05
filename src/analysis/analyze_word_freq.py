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

# Track overused words with their timestamps
overused_words = {}

# Parameters for detecting overuse
TIME_WINDOW = 10  # seconds
FREQUENCY_THRESHOLD = 3  # word appears >= this in the time window

for word, instances in timestamps.items():
    word_lower = word.lower().strip()
    if not re.match(r"^[a-zA-Z']+$", word_lower):
        continue

    # Get list of timestamps
    time_stamps = sorted([t["start"] for t in instances])
    if len(time_stamps) < FREQUENCY_THRESHOLD:
        continue

    # Slide a window across timestamps to check density
    for i in range(len(time_stamps) - FREQUENCY_THRESHOLD + 1):
        window = time_stamps[i:i+FREQUENCY_THRESHOLD]
        if window[-1] - window[0] <= TIME_WINDOW:
            overused_words[word_lower] = {
                "type": "content",
                "count": len(instances),
                "instances": instances
            }
            break

# Save to report
with open(report_path, "w") as f:
    json.dump(overused_words, f, indent=4)

print(f"✅ Overused Words Report saved to: {report_path}")