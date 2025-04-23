import json
from transformers import pipeline
import argparse
import re
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Detect profanity and unprofessional language")
parser.add_argument("timestamp_path", help="Path to timestamps JSON file")
parser.add_argument("transcription_path", help="Path to transcription text file")
parser.add_argument("--report_path", default="src/tests/results/profanity_report.json", help="Output path for the report JSON")
args = parser.parse_args()

# Assign argument values to variables
transcription_path = args.transcription_path
timestamp_data_path = args.timestamp_path
report_path = args.report_path

# Ensure output directory exists
os.makedirs(os.path.dirname(report_path), exist_ok=True)

# Load pre-trained Hugging Face text classification models
# These models are used to detect offensive/profane language
profanity_pipe = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target")
offensive_pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-offensive")

# Define regex patterns to catch informal, slang, or unprofessional expressions
# These are not necessarily profane, but are discouraged in formal speech
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

# Load full transcript (not used here, but included for completeness)
with open(transcription_path, "r") as f:
    transcribed_text = f.read()

# Load the timestamped words from JSON
with open(timestamp_data_path, "r") as f:
    timestamps = json.load(f)

# Initialize list to store flagged word results
filtered_results = []

# Loop through each word entry from the timestamps
for entry in timestamps:
    word = entry["word"]
    word_lower = word.lower().strip()
    start = entry["start_time"]
    end = entry["end_time"]
    
    # First check for hardcoded unprofessional words using regex
    if any(re.search(pattern, word_lower) for pattern in UNPROFESSIONAL_PATTERNS):
        category = "unprofessional"
        confidence = 0.99 # high confidence since it's a direct match
    else:
        # Run the word through the two language classification models
        # Output: profanity_pipe("damn") -> [{'label': 'profanity', 'score': 0.892}]
        result1 = profanity_pipe(word_lower)[0]
        result2 = offensive_pipe(word_lower)[0]
        
        # Check if the models detect profanity or offensive tone with high confidence
        if result1["label"] == "profanity" and result1["score"] > 0.6:
            category = "profanity"
            confidence = result1["score"]
        elif result2["label"] == "offensive" and result2["score"] > 0.6:
            category = "offensive"
            confidence = result2["score"]
        else:
            continue # skip if not flagged by any model
        
    # Append the flagged word and its metadata to the results
    filtered_results.append({
        "word": word,
        "category": category,
        "start_time": start,
        "end_time": end,
        "confidence": confidence
    })

# Save the final list of flagged words to a JSON report
with open(report_path, "w") as f:
    json.dump(filtered_results, f, indent=4)

# Confirmation message
print(f"✅ Profanity Report saved to: {report_path}")