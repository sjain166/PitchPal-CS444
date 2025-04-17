# •	librosa to load audio,
# •	torch for model inference,
# •	transformers to load a pretrained emotion model,
# •	softmax to convert raw scores into probabilities.
import torch
import librosa
import json
import argparse
import os
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from scipy.special import softmax

# •	audio_path → path to the .wav file.
# •	timestamp_path → JSON file with word-level timestamps.
# •	output_path → where the emotion results should be saved.
parser = argparse.ArgumentParser(description="Analyze emotion from audio using timestamp ranges.")
parser.add_argument("audio_path", help="Path to the input .wav file")
parser.add_argument("timestamp_path", help="Path to the timestamps JSON file (sentence-level)")
parser.add_argument("--output_path", default="./tests/results/emotion_analysis.json", help="Path to save results")
args = parser.parse_args()

# Loads the audio file at a fixed 16kHz sample rate — required by the emotion model.
audio, sr = librosa.load(args.audio_path, sr=16000)

# Loads word-level timestamps (start/end time for each word spoken).
with open(args.timestamp_path, "r") as f:
    timestamps = json.load(f)

# • Loads the pre-trained emotion detection model and its corresponding feature extractor.
# •	The model is from Hugging Face and trained on RAVDESS dataset.
# •	Moved to GPU if available.
# CUDA- Nvidia chip (not supported on Mac)
device = "cuda" if torch.cuda.is_available() else "cpu" # can use 'mps' (Metal Performance Shaders) for Apple
model_name = "superb/hubert-large-superb-er"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)
# Use "model.train()" for training.
model.eval() # It tells PyTorch that you’re not training, you’re evaluating the model.

# Prepares to convert word-level timestamps into larger chunks (around 10 seconds long) for analysis.
chunks = []
curr_chunk = []
start_time = 0.0

# Groups words into ~5s chunks based on timing difference.
# • Each chunk represents a segment of continuous speech that will be fed into the emotion model.
for word in timestamps:
    if not curr_chunk:
        start_time = word["start_time"]
    curr_chunk.append(word)
    if word["end_time"] - start_time >= 5.0:
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

# Iterates over each chunk and extracts that part of the audio using sample indices.
results = []

for segment in chunks:
    start = segment["start_time"]
    end = segment["end_time"]
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    chunk = audio[start_sample:end_sample]
    
    # Skips less than 5 seconds segments (likely too short to analyze meaningfully).
    if len(chunk) < 5*16000: # 16000 sampling rate
        continue
    
    # Processes audio through the model to get raw prediction scores (logits).
    # Returns a PyTorch-compatible tensor (return_tensors="pt").
    inputs = extractor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        # Sends inputs to the appropriate device (CPU/GPU).
        # **inputs unpacks the dictionary into keyword args
        # •	The model outputs a logits tensor:
        # •	These are raw scores (before softmax) for each emotion class.
        logits = model(**inputs.to(device)).logits
        
    # Converts logits to probabilities using softmax and extracts the top-predicted emotion and its confidence score.
    # • logits.cpu() → moves it to CPU (safe for NumPy).
	# •	numpy() → converts it into a NumPy array.
    #  softmax(...) → converts raw logits into probabilities for each emotion class.
    scores = softmax(logits.cpu().numpy()[0])
    # •	argmax() returns the index of the highest score (i.e., most confident emotion prediction).
    top_idx = scores.argmax()
    # Uses the model’s built-in mapping of index → label.
    label = model.config.id2label[top_idx]
    confidence = float(scores[top_idx])

    # Skips segments that are either:
	# •	Low confidence (< 51%)
	# •	Positive/neutral emotions that don’t need feedback (like “happy” or "neutral").
    if confidence < 0.51 or label.lower() in ["neu", "hap"]:
        continue
    
    # Assigns feedback strength based on confidence:
	# •	recommended feedback if model is very sure
	# •	possible feedback otherwise
    comment = "recommended feedback" if confidence >= 0.9 else "possible feedback"
    
    # Stores the final result for the chunk, including timestamps, emotion, and feedback strength.
    results.append({
        "start_time": round(start, 2),
        "end_time": round(end, 2),
        "predicted_emotion": label,
        "confidence": round(confidence, 3),
        "comment": comment
    })

# Saves the results to a JSON file and prints a confirmation.
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
with open(args.output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Emotion analysis saved to {args.output_path}")