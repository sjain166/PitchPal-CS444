import json
import argparse
import os
from collections import defaultdict

# --- Command-line argument parsing ---
parser = argparse.ArgumentParser(description="Detect filler words from transcription timestamps.")
parser.add_argument("timestamp_path", help="Path to timestamps JSON file")
parser.add_argument("--report_path", default="./tests/results/filler_report.json", help="Output path for the filler report JSON")
args = parser.parse_args()

timestamp_path = args.timestamp_path
report_path = args.report_path

# Ensure the output directory exists
os.makedirs(os.path.dirname(report_path), exist_ok=True)

# Load timestamped words
with open(timestamp_path, "r") as f:
    timestamps = json.load(f)

# Define set of filler tokens (normalized to lowercase, trimmed)
FILLER_WORDS = {"[um]", "[uh]", "[ah]", "[hmm]", "[erm]"}

# Aggregate all occurrences per filler word
word_timestamps = defaultdict(list)
for entry in timestamps:
    word = entry.get("word", "").strip()
    normalized = word.lower().strip()
    if normalized in FILLER_WORDS:
        word_timestamps[word].append({
            "start_time": round(entry.get("start_time", 0.0), 2),
            "end_time": round(entry.get("end_time", 0.0), 2)
        })

# Build the final report list with grouped timestamps
report = []
for word, ts_list in word_timestamps.items():
    report.append({
        "word": word,
        "timestamps": ts_list
    })

# Save the report as JSON
with open(report_path, "w") as f:
    json.dump(report, f, indent=4)

print(f"✅ Filler Report saved to: {report_path}")