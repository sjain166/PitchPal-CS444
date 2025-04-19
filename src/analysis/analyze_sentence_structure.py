# Keep your original ArgumentParser exactly as-is
import argparse
import json
import spacy
from transformers import pipeline

parser = argparse.ArgumentParser(description="Analyze sentence structure and relevance.")
parser.add_argument("timestamp_path", help="Path to timestamps JSON file")
parser.add_argument("transcription_path", help="Path to transcription .txt file")
parser.add_argument("--output_path", default="./tests/results/sentence_structure_report.json", help="Path to profanity report JSON file")
args = parser.parse_args()

# Load SpaCy and classifier
nlp = spacy.load("en_core_web_sm")
classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-base")

# ---- Load Transcription ----
with open(args.transcription_path, "r") as f:
    transcription = f.read().strip()

# ---- Break into Sentences ----
doc = nlp(transcription)
raw_sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

sentences = []
buffer = ""

for sent in raw_sentences:
    word_count = len(sent.split())
    ends_with_punct = sent.endswith((".", "!", "?"))

    if word_count < 7 and not ends_with_punct:
        buffer += " " + sent
    else:
        buffer += " " + sent
        sentences.append(buffer.strip())
        buffer = ""

if buffer.strip():
    sentences.append(buffer.strip())

# ---- Define Labels ----
labels = [
  "Speaker introduces themselves",
  "Speaker mentions their education",
  "Speaker describes technical experience",
  "Speaker discusses future career goals or interests",
  "Speaker highlights achievements or personal strengths",
  "Speaker requests an opportunity like an internship",
  "Speaker explains their motivation or values",
  "Speaker makes an off-topic or confusing statement"
]

# ---- Classify Each Sentence ----
results = []
for sentence in sentences:
    out = classifier(sentence, labels)
    label = out["labels"][0]
    raw_score = out["scores"][0]

    # Invert the confidence so that high means more relevant
    relevance = round(1 - raw_score, 2)
    if label == "Speaker makes an off-topic or confusing statement":
        results.append({
            "sentence": sentence,
            "relevance": relevance,
            "feedback": "Off-topic statement"
        })
    
# ---- Save to JSON ----
with open(args.output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Saved report to {args.output_path}")