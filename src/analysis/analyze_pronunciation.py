import argparse
import os
import json
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# --- Parse arguments ---
parser = argparse.ArgumentParser(description="Analyze pronunciation quality using CMU Pronouncing Dictionary")
parser.add_argument("audio_path", help="Path to the input .wav file")
parser.add_argument("timestamp_path", help="Path to timestamp.json (word-level Whisper output)")
parser.add_argument("--output_path", default="./tests/results/pronunciation_report.json", help="Where to save the report")
args = parser.parse_args()

# --- Load audio ---
audio, sr = librosa.load(args.audio_path, sr=16000)

# --- Load ASR model and processor ---
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).eval()

# --- Load timestamp JSON ---
with open(args.timestamp_path, "r") as f:
    word_entries = json.load(f)

# --- Analyze pronunciation ---
report = []

for entry in word_entries:
    word = entry["word"].strip(".,?!").lower()
    start = entry["start_time"]
    end = entry["end_time"]
    
    print(f"🔍 Checking word: '{word}' from {start:.2f}s to {end:.2f}s")

    if not word.isalpha():
        continue

    start_sample = int(start * sr)
    end_sample = int(end * sr)
    chunk = audio[start_sample:end_sample]

    if len(chunk) < 500:
        continue

    inputs = processor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0].lower().strip()

    print(f"ASR output: '{transcription}'")
    if word not in transcription.split():
        print("❌ Word not found in ASR output — possible mispronunciation.")
        report.append({
            "word": word,
            "start_time": start,
            "end_time": end,
            "predicted_transcript": transcription,
            "note": "Possible mispronunciation (not detected by ASR)"
        })
    else:
        print("    ✅ Word appears correctly.")

# --- Save results ---
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
with open(args.output_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"✅ Pronunciation report saved to {args.output_path}")