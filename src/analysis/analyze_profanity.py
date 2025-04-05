import json
from transformers import pipeline
import argparse
import re
import os

# Set up argument parser for command line inputs
parser = argparse.ArgumentParser(description="Detect profanity and unprofessional language")
parser.add_argument("timestamp_path", help="Path to timestamps JSON file")
parser.add_argument("transcription_path", help="Path to transcription text file")
parser.add_argument("--report_path", default="./tests/results/profanity_report.json", help="Output path for the report JSON")
args = parser.parse_args()

# Assign command line arguments to variables
transcription_path = args.transcription_path
timestamp_data_path = args.timestamp_path
report_path = args.report_path

# Create directory for report if it doesn't exist
os.makedirs(os.path.dirname(report_path), exist_ok=True)

# Load language analysis models for detecting profanity and offensive language
profanity_pipe = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target")
offensive_pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-offensive")

# Define a set of unprofessional words that are not profane but unsuitable for a pitch
UNPROFESSIONAL_PATTERNS = [
    r"\bgonn?a\b",
    r"\bwann?a\b",
    r"\bgotta\b",
    r"\bain['’`]?t\b",
    r"\bdude\b",
    r"\bbro+\b",
    r"\byo+\b",
    r"\bnah+\b",
    r"\bbruh\b",
    r"\bsup\b",
    r"\bcuz\b",
    r"\bshit\b", r"\bdamn\b", r"\bcrap\b", r"\bhell\b", r"\bfreakin\b", r"\bfrick\b", r"\bsucks\b", r"\bbloody\b",
    r"\bkinda\b", r"\bsorta\b",
    r"\bthingy\b",
    r"\bwhatev[ae]r\b",
    r"\blowkey\b",
    r"\bhighkey\b",
    r"\bnigg(a|ah|as|az|er|ers|uh|urs)?\b"
]

# Load the transcribed text from the specified file
with open(transcription_path, "r") as f:
    transcribed_text = f.read()

# Load word-level timestamps from the specified JSON file
with open(timestamp_data_path, "r") as f:
    timestamps = json.load(f)

# Initialize a list to hold filtered results of profanity and unprofessional speech
filtered_results = []

# Analyze each word from the transcription against predefined categories
for word, time_data_list in timestamps.items():
    word_lower = word.lower().strip()  # Normalize the word to lowercase and remove whitespace

    # Check if the word is in the set of unprofessional words
    if any(re.search(pattern, word_lower) for pattern in UNPROFESSIONAL_PATTERNS):
        category = "unprofessional"
        confidence = 0.99

    # Use AI models to check for profanity or offensive words
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

    # Store the results for each occurrence of the word along with its timestamps
    for time_data in time_data_list:
        filtered_results.append({
            "word": word,
            "category": category,
            "start": time_data["start"],
            "end": time_data["end"],
            "confidence": confidence
        })

# Save the filtered analysis report to the specified JSON file
with open(report_path, "w") as f:
    json.dump(filtered_results, f, indent=4)

# Print summary of the analysis
print(f"✅ Profanity Report saved to: {report_path}")