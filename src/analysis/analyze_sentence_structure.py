import re
import spacy # spacy for sentence tokenization and grammar parsing
import os
import json
import argparse
from sentence_transformers import SentenceTransformer, util

# Load SpaCy English model for sentence tokenization and dependency parsing
nlp = spacy.load("en_core_web_sm")

# Sets up the script to receive the transcription text, timestamps, and profanity report.
parser = argparse.ArgumentParser(description="Analyze sentence structure and relevance.")
parser.add_argument("timestamp_path", help="Path to timestamps JSON file")
parser.add_argument("transcription_path", help="Path to transcription .txt file")
parser.add_argument("profanity_report_path", help="Path to profanity report JSON file")
output_path = "./tests/results/sentence_analysis_report.json"
args = parser.parse_args()

# These values define the cutoff for giving sentence-level feedback based on relevance.
LOW_RELEVANCE_THRESHOLD = 0.15
WEAK_RELEVANCE_THRESHOLD = 0.25

# Load the transcription, word-level timestamps, and the profanity analysis report.
with open(args.transcription_path, "r") as f:
    raw_text = f.read()

with open(args.timestamp_path, "r") as f:
    timestamps = json.load(f)

with open(args.profanity_report_path, "r") as f:
    profanity_data = json.load(f)

# Cleans up disfluencies and spacing in the transcript to prepare for parsing.
cleaned_text = re.sub(r"\[(UM|UH)\]", "", raw_text)
cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

# Split into tokens using SpaCy
doc = nlp(cleaned_text)

# Replace SpaCy's sentence segmentation with dynamic chunks of 10–15 words
tokens = [token for token in doc if token.is_alpha or token.text in [".", "!", "?"]]
chunk_size = 12
step_size = 10

model = SentenceTransformer('all-MiniLM-L6-v2')
meaningful_sentences = []
sentence_time_ranges = []

for i in range(0, len(tokens), step_size):
    chunk = tokens[i:i + chunk_size]
    # This reconstructs the actual text from the selected chunk of tokens.
    sent_text = " ".join([tok.text for tok in chunk]).strip()

    # These checks use SpaCy’s parsed dependency tree and POS tags:
	# •	nsubj, nsubjpass: subject or passive subject.
	# •	VERB: verb in the sentence.
    has_subject = any(tok.dep_ in ("nsubj", "nsubjpass") for tok in chunk)
    has_verb = any(tok.pos_ == "VERB" for tok in chunk)
    if not (has_subject and has_verb):
        continue  # Skip if not meaningful

    sent_words = [t.text.lower().strip(".,!?") for t in chunk if t.is_alpha]
    matched_times = [entry for entry in timestamps if entry["word"].lower() in sent_words]
    matched_times.sort(key=lambda x: x["start_time"])

    if matched_times:
        start_time = matched_times[0]["start_time"]
        end_time = matched_times[min(8, len(matched_times) - 1)]["end_time"]
    else:
        start_time = end_time = None

    meaningful_sentences.append(sent_text)
    sentence_time_ranges.append((start_time, end_time))

# Compare user’s sentences to professionally phrased reference sentences using cosine similarity.
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
# Compute the highest similarity score for each user sentence.
pitch_relevance = [
    max(util.cos_sim(e, ref).item() for ref in reference_embeddings)
    for e in embeddings
]

# Classify each sentence as weak, borderline, or good based on its similarity score.
def categorize_relevance(score):
    if score < LOW_RELEVANCE_THRESHOLD:
        return "Recommended feedback"
    elif score < WEAK_RELEVANCE_THRESHOLD:
        return "Might take a look at feedback"
    else:
        return None

# Identify profanity or slang words that appear during the sentence’s time span.
def match_profanity(start, end):
    matched = set()
    for entry in profanity_data:
        if start is None or end is None:
            continue
        if entry["start_time"] <= end and entry["end_time"] >= start:
            if entry["category"] == "offensive" or entry["category"] == "profanity":
                matched.add("Profanity alert")
            elif entry["category"] == "unprofessional":
                matched.add("Possible unprofessional tone")
    return list(matched)

# For each sentence, combine grammar/semantic feedback with profanity flags and create a final comment.
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

# Save the analysis to disk in a structured JSON format.
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(summary_output, f, indent=4)
    
print(f"✅ Profanity Report saved to: {output_path}")