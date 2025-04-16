# •	argparse for handling command-line arguments,
# •	librosa for loading audio,
# •	transformers to use Wav2Vec2 ASR model,
# •	torch for inference,
# •	json and os for file I/O.
import argparse
import os
import json
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# --- Parse arguments ---
# •	The original audio file,
# •	Word-level timestamp JSON file (from Whisper),
# •	Output path where pronunciation issues will be saved.
parser = argparse.ArgumentParser(description="Analyze pronunciation quality using CMU Pronouncing Dictionary")
parser.add_argument("audio_path", help="Path to the input .wav file")
parser.add_argument("timestamp_path", help="Path to timestamp.json (word-level Whisper output)")
parser.add_argument("--output_path", default="./tests/results/pronunciation_report.json", help="Where to save the report")
args = parser.parse_args()

# --- Load audio ---
#  Load the audio file at 16kHz — the required sample rate for Wav2Vec2 models.
audio, sr = librosa.load(args.audio_path, sr=16000)

# --- Load ASR model and processor ---
# •	Wav2Vec2Processor tokenizes and processes audio,
# •	Wav2Vec2ForCTC transcribes audio using Connectionist Temporal Classification (CTC),
# •	Set to evaluation mode with .eval() (not training).

model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).eval()

# --- Load timestamp JSON ---
# Load the timestamped words, each with start and end times for isolated analysis.
with open(args.timestamp_path, "r") as f:
    word_entries = json.load(f)

# --- Analyze pronunciation ---
# Iterate through each word, cleaning it up (e.g., punctuation), and prepare for ASR verification using audio slice.
report = []

for entry in word_entries:
    word = entry["word"].strip(".,?!").lower()
    start = entry["start_time"]
    end = entry["end_time"]
    
    print(f"🔍 Checking word: '{word}' from {start:.2f}s to {end:.2f}s")
    # Skip non-word tokens like pauses or symbols (e.g., “[noise]”).
    if not word.isalpha():
        continue
    
    # Extract the audio chunk that corresponds to just this word, based on timestamps.
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    chunk = audio[start_sample:end_sample]
    
    # Skip very short audio clips (likely noise or silence).
    if len(chunk) < 500:
        continue
    
    # •	Preprocess the audio with the processor,
	# •	Feed it into the model to get logits (raw outputs),
	# •	Use argmax to find the most probable token ID,
	# •	Decode the prediction into a word/phrase.
    inputs = processor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0].lower().strip()
        
    # Compare expected word to predicted ASR result:
	# •	If the expected word is missing from the ASR prediction, flag it as potentially mispronounced.
	# •	Otherwise, print confirmation.
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
# Write all flagged cases to a JSON report for review, and confirm completion.
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
with open(args.output_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"✅ Pronunciation report saved to {args.output_path}")