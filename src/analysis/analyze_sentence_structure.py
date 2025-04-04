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
args = parser.parse_args()
input_file = args.transcription_path

LOW_RELEVANCE_THRESHOLD = 0.15
WEAK_RELEVANCE_THRESHOLD = 0.25

with open(input_file, "r") as f:
    raw_text = f.read()

# Remove filler tokens like [UM], [UH]
cleaned_text = re.sub(r"\[(UM|UH)\]", "", raw_text)

# Fix spacing
cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

# Sentence segmentation using spaCy
doc = nlp(cleaned_text)
candidate_sentences = [sent.text.strip() for sent in doc.sents]

# Filter: Keep only sentences with at least one subject and one verb
meaningful_sentences = []
for sent in doc.sents:
    has_subject = any(tok.dep_ in ("nsubj", "nsubjpass") for tok in sent)
    has_verb = any(tok.pos_ == "VERB" for tok in sent)
    if has_subject and has_verb:
        meaningful_sentences.append(sent.text.strip())

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Good example elevator pitch sentences
good_examples = [
    "I am currently studying Cloud Computing, where I'm working on AWS-based projects.",
    "In my Computer Vision course, I'm developing CNNs from scratch for image recognition.",
    "I'm leading a distributed systems project focused on efficient peer-to-peer networking.",
    "My coursework includes writing weekly research paper reviews, which has sharpened my critical thinking."
]
good_embeddings = model.encode(good_examples)

# Compute embeddings
embeddings = model.encode(meaningful_sentences)

# Sentence-to-sentence coherence
coherence_scores = []
for i, emb in enumerate(embeddings):
    rest = [e for j, e in enumerate(embeddings) if j != i]
    avg_similarity = sum(util.cos_sim(emb, r).item() for r in rest) / len(rest)
    coherence_scores.append(avg_similarity)

# Elevator pitch relevance
elevator_pitch_prompt = "I am answering 'Introduce Yourself' for Interviews."
pitch_embedding = model.encode(elevator_pitch_prompt)
pitch_relevance = [util.cos_sim(e, pitch_embedding).item() for e in embeddings]

def suggest_improvement(sentence, relevance_score, sentence_embedding):
    if relevance_score >= WEAK_RELEVANCE_THRESHOLD:
        return None

    # Find the closest good sentence
    similarities = [util.cos_sim(sentence_embedding, good_emb).item() for good_emb in good_embeddings]
    best_idx = similarities.index(max(similarities))
    closest_example = good_examples[best_idx]

    if relevance_score < LOW_RELEVANCE_THRESHOLD:
        return (
            f"⚠️ This sentence feels off-topic. Try modeling it like: \"{closest_example}\""
        )
    else:
        return (
            f"✅ Close! You can improve this by aligning more with something like: \"{closest_example}\""
        )

# Print analysis
# for i, sentence in enumerate(meaningful_sentences):
#     print(f"\n🔹 Sentence {i+1}: {sentence}")
#     print(f"   - Coherence Score: {coherence_scores[i]:.2f}")
#     print(f"   - Elevator Pitch Relevance: {pitch_relevance[i]:.2f}")
#     suggestion = suggest_improvement(sentence, pitch_relevance[i], embeddings[i])
#     if suggestion:
#         print(f"   💡 Suggestion: {suggestion}")
summary_output = []
for i, sentence in enumerate(meaningful_sentences):
    suggestion = suggest_improvement(sentence, pitch_relevance[i], embeddings[i])
    summary_output.append({
        "sentence": sentence,
        "coherence_score": round(coherence_scores[i], 2),
        "relevance_score": round(pitch_relevance[i], 2),
        "suggestion": suggestion or "✅ Looks good!"
    })

# Save to JSON
output_path = "./tests/results/sentence_analysis_report.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(summary_output, f, indent=4)