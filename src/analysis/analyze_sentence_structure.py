import re
import spacy
import os
import json
import argparse
from sentence_transformers import SentenceTransformer, util
nlp = spacy.load("en_core_web_sm")

parser = argparse.ArgumentParser(description="Analyze sentence structure and relevance.")
parser.add_argument("timestamp_path", help="Path to timestamps JSON file")
parser.add_argument("transcription_path", help="Path to transcription .txt file")
parser.add_argument("profanity_report_path", help="Path to profanity report JSON file")
output_path = "./tests/results/sentence_analysis_report.json"
args = parser.parse_args()

LOW_RELEVANCE_THRESHOLD = 0.15
WEAK_RELEVANCE_THRESHOLD = 0.25

with open(args.transcription_path, "r") as f:
    raw_text = f.read()

with open(args.timestamp_path, "r") as f:
    timestamps = json.load(f)

with open(args.profanity_report_path, "r") as f:
    profanity_data = json.load(f)

cleaned_text = re.sub(r"\[(UM|UH)\]", "", raw_text)
cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

doc = nlp(cleaned_text)

model = SentenceTransformer('all-MiniLM-L6-v2')

meaningful_sentences = []
sentence_time_ranges = []

for sent in doc.sents:
    sent_text = sent.text.strip()
    has_subject = any(tok.dep_ in ("nsubj", "nsubjpass") for tok in sent)
    has_verb = any(tok.pos_ == "VERB" for tok in sent)
    if not (has_subject and has_verb):
        continue

    sent_words = [t.text.lower().strip(".,!?") for t in sent if t.is_alpha]
    times = [t["start"] for word in sent_words if word in timestamps for t in timestamps[word]]
    if times:
        start_time = min(times)
        end_time = max([t["end"] for word in sent_words if word in timestamps for t in timestamps[word]])
    else:
        start_time = end_time = None

    meaningful_sentences.append(sent_text)
    sentence_time_ranges.append((start_time, end_time))

embeddings = model.encode(meaningful_sentences)

reference_sentences = [
    "I completed my degree in my field",
    "I recently graduated with a background in a subject area",
    "I'm currently working as a professional in my industry",
    "My interests lie in applying my skills to real-world problems",
    "I'm passionate about solving challenges in my domain",
    "I transitioned from my studies to full-time work",
    "I earned my degree with a focus on my area of interest",
    "I have experience conducting research and projects",
    "I'm pursuing further education in a specialized area",
    "My background includes academic and practical experiences",
    "I have worked on various interdisciplinary projects",
    "I completed internships related to my career goals",
    "I’ve led or participated in collaborative team efforts",
    "I'm passionate about presenting information effectively",
    "I’ve worked with diverse teams on real-world challenges",
    "I'm excited about applying my skills to create impact"
]
reference_embeddings = model.encode(reference_sentences)
pitch_relevance = [
    max(util.cos_sim(e, ref).item() for ref in reference_embeddings)
    for e in embeddings
]

def categorize_relevance(score):
    if score < LOW_RELEVANCE_THRESHOLD:
        return "Recommended feedback"
    elif score < WEAK_RELEVANCE_THRESHOLD:
        return "Might take a look at feedback"
    else:
        return None

def match_profanity(start, end):
    matched = set()
    for entry in profanity_data:
        if start is None or end is None:
            continue
        if entry["start"] <= end and entry["end"] >= start:
            if entry["category"] == "offensive" or entry["category"] == "profanity":
                matched.add("Profanity alert")
            elif entry["category"] == "unprofessional":
                matched.add("Possible unprofessional tone")
    return list(matched)

summary_output = []
for i, sentence in enumerate(meaningful_sentences):
    rel_score = round(pitch_relevance[i], 2)
    base_comment = categorize_relevance(rel_score)
    prof_comments = match_profanity(*sentence_time_ranges[i])
    combined_comment = " | ".join(filter(None, [base_comment] + prof_comments)) or "Looks good!"

    summary_output.append({
        "sentence": sentence,
        "timestamp": {
            "start_time": sentence_time_ranges[i][0],
            "end_time": sentence_time_ranges[i][1]
        },
        "relevance_score": rel_score,
        "comment": combined_comment
    })

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(summary_output, f, indent=4)