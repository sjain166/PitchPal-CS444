import re
import spacy
import os
import json
import argparse
from transformers import pipeline

# Load SpaCy English model for sentence segmentation
nlp = spacy.load("en_core_web_sm")

# Keep your original ArgumentParser exactly as-is
parser = argparse.ArgumentParser(description="Analyze sentence structure and relevance.")
parser.add_argument("timestamp_path", help="Path to timestamps JSON file")
parser.add_argument("transcription_path", help="Path to transcription .txt file")
parser.add_argument("sentence_structure_path", help="Path to profanity report JSON file")  # unused here
args = parser.parse_args()

# Load inputs
with open(args.transcription_path, "r") as f:
    raw_text = f.read().strip()

with open(args.timestamp_path, "r") as f:
    timestamps = json.load(f)

# Pre‑compile a little helper to strip punctuation for matching
def clean_word(w):
    return re.sub(r"[^\w']+", "", w.lower())

# Build a zero‑shot classifier once
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

candidate_labels = ["relevant", "irrelevant"]
hypothesis_template = "This sentence is {} for an elevator pitch."

IRRELEVANT_THRESHOLD = 0.5

results = []

# 1) Sentence segmentation
doc = nlp(raw_text)
for sent in doc.sents:
    sent_text = sent.text.strip()
    if not sent_text:
        continue

    # 2) Align timestamps
    sent_tokens = [clean_word(tok.text) for tok in sent if tok.text.strip()]
    matched = [
        ts for ts in timestamps
        if clean_word(ts["word"]) in sent_tokens
    ]
    matched = sorted(matched, key=lambda x: x["start_time"])
    if matched:
        start_time = matched[0]["start_time"]
        end_time = matched[-1]["end_time"]
    else:
        start_time = None
        end_time = None

    # 3) Zero‑shot classification
    out = classifier(
        sent_text,
        candidate_labels=candidate_labels,
        hypothesis_template=hypothesis_template,
    )
    # grab the “irrelevant” score
    irrelevant_score = out["scores"][out["labels"].index("irrelevant")]
    flagged = irrelevant_score > IRRELEVANT_THRESHOLD

    # 4) Collect
    results.append({
        "sentence": sent_text,
        "start_time": start_time,
        "end_time": end_time,
        "irrelevant_score": round(irrelevant_score, 2),
        "flagged": flagged
    })

# 5) Write JSON
dirpath = os.path.dirname(args.sentence_structure_path)
if dirpath and not os.path.exists(dirpath):
    os.makedirs(dirpath, exist_ok=True)

with open(args.sentence_structure_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"✅ Sentence‐structure analysis saved to: {args.sentence_structure_path}")