import torch
import librosa
import json
import argparse
import os
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from scipy.special import softmax

# Load args
parser = argparse.ArgumentParser(description="Analyze emotion from audio using timestamp ranges.")
parser.add_argument("audio_path", help="Path to the input .wav file")
parser.add_argument("timestamp_path", help="Path to the timestamps JSON file (sentence-level)")
parser.add_argument("--output_path", default="./tests/results/emotion_analysis.json", help="Path to save results")
args = parser.parse_args()

# Load audio
audio, sr = librosa.load(args.audio_path, sr=16000)

# Load timestamp ranges
with open(args.timestamp_path, "r") as f:
    timestamps = json.load(f)

# Load model + feature extractor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Wiam/wav2vec2-lg-xlsr-en-speech-emotion-recognition-finetuned-ravdess-v8"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)
model.eval()

# Convert word-level timestamps into 10s chunks
chunks = []
curr_chunk = []
start_time = 0.0

for word in timestamps:
    if not curr_chunk:
        start_time = word["start_time"]
    curr_chunk.append(word)
    if word["end_time"] - start_time >= 10.0:
        chunks.append({
            "start_time": start_time,
            "end_time": word["end_time"]
        })
        curr_chunk = []

if curr_chunk:
    chunks.append({
        "start_time": start_time,
        "end_time": curr_chunk[-1]["end_time"]
    })

# Predict emotion for each chunk
results = []

for segment in chunks:
    start = segment["start_time"]
    end = segment["end_time"]

    start_sample = int(start * sr)
    end_sample = int(end * sr)
    chunk = audio[start_sample:end_sample]

    if len(chunk) < 1000:
        continue

    inputs = extractor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs.to(device)).logits

    scores = softmax(logits.cpu().numpy()[0])
    top_idx = scores.argmax()
    label = model.config.id2label[top_idx]
    confidence = float(scores[top_idx])

    # Skip low-confidence or positive emotions
    if confidence < 0.51 or label.lower() in ["calm", "neutral", "confident", "happy"]:
        continue

    comment = "recommended feedback" if confidence >= 0.9 else "possible feedback"

    results.append({
        "start_time": round(start, 2),
        "end_time": round(end, 2),
        "predicted_emotion": label,
        "confidence": round(confidence, 3),
        "comment": comment
    })

# Save results
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
with open(args.output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Emotion analysis saved to {args.output_path}")