import json
import re
from transformers import pipeline
import argparse
import os

parser = argparse.ArgumentParser(description="Detect profanity and unprofessional language")
parser.add_argument("timestamp_path", help="Path to timestamps JSON file")
parser.add_argument("transcription_path", help="Path to transcription text file")
parser.add_argument("--report_path", default="./tests/results/profanity_report.json", help="Output path for the report JSON")
args = parser.parse_args()
transcription_path = args.transcription_path
timestamp_data_path = args.timestamp_path
report_path = args.report_path
os.makedirs(os.path.dirname(report_path), exist_ok=True)

print("🔄 Loading Language Analysis Models...")
# Detects racial slurs
profanity_pipe = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target")
# Filters out offensive words
offensive_pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-offensive")

# Define a List of Words That AI Models May Miss
HARDCODED_PROFANITY = [
    r"nigg[a|er|uh|ah|as]*",
]

# Define Additional "Unprofessional" Words (Non-profane, but not suitable for a pitch)
UNPROFESSIONAL_WORDS = {
    "gonna", "wanna", "ain’t", "dude", "bro", "yo", "shit", "damn", "crap", "hell", "kinda", "sorta"
}

# Load Transcribed Text
with open(transcription_path, "r") as f:
    transcribed_text = f.read()

# Load Word-Level Timestamps
with open(timestamp_data_path, "r") as f:
    timestamps = json.load(f)

# Analyze Each Word for Profanity & Unprofessional Speech
filtered_results = []
for word, time_data_list in timestamps.items():
    word_lower = word.lower().strip()  # Normalize word

    # Check Against Predefined List of Unprofessional Words
    if word_lower in UNPROFESSIONAL_WORDS:
        category = "unprofessional"
        confidence = 0.99

    # Check Hardcoded Profanity (Regex Match)
    elif any(re.match(pattern, word_lower) for pattern in HARDCODED_PROFANITY):
        category = "profanity"
        confidence = 1.0

    # Check AI-Based Models for Profanity/Offensive Words
    else:
        result1 = profanity_pipe(word_lower)[0]
        result2 = offensive_pipe(word_lower)[0]

        if result1["label"] == "profanity" and result1["score"] > 0.6:
            category = "profanity"
            confidence = result1["score"]
        elif result2["label"] == "offensive" and result2["score"] > 0.6:
            category = "offensive"
            confidence = result2["score"]
        else:
            continue

    # Store Results for Each Occurrence of the Word
    for time_data in time_data_list:
        filtered_results.append({
            "word": word,
            "category": category,
            "start": time_data["start"],
            "end": time_data["end"],
            "confidence": confidence
        })

# Save Filtered Words Analysis Report
with open(report_path, "w") as f:
    json.dump(filtered_results, f, indent=4)

# Print Summary
print("✅ Language Analysis Complete!")
print(f"🚨 {len(filtered_results)} inappropriate words detected.")
print(f"🔍 Report saved to: {report_path}")

# Print Results
for instance in filtered_results:
    print(f"🚨 {instance['category'].upper()} | {instance['word']} at {instance['start']}s (Confidence: {instance['confidence']:.2f})")