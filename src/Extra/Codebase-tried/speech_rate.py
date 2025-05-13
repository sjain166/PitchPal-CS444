# !pip install transformers torchaudio accelerate

from transformers import pipeline
import numpy as np
import torch
import torchaudio
import os

# ---- SETTINGS ----
AUDIO_PATH = "/Users/aryangupta/Desktop/UIUC/CURRENT/CS-444/PitchPal-CS444/data/fast.wav"  # Upload your file to Colab

# ---- LOAD PIPELINE ----
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    chunk_length_s=30,
    return_timestamps="word",
    device=0 if torch.cuda.is_available() else -1
)

# ---- RUN TRANSCRIPTION ----
print("🔁 Transcribing...")
result = asr(AUDIO_PATH)

# ---- ANALYZE SPEECH RATE ----
# Filter out words with valid timestamps
valid_words = [
    w for w in result['chunks']
    if isinstance(w.get('timestamp'), (list, tuple)) and
       len(w['timestamp']) == 2 and
       w['timestamp'][0] is not None and
       w['timestamp'][1] is not None
]

if len(valid_words) < 2:
    print("❌ Not enough valid timestamps to compute speech rate.")
else:
    start_time = valid_words[0]['timestamp'][0]
    end_time = valid_words[-1]['timestamp'][1]
    total_duration_min = (end_time - start_time) / 60.0
    word_count = len(valid_words)
    wpm = word_count / total_duration_min

    print(f"🗣️ Total Words: {word_count}")
    print(f"⏱️ Duration: {end_time - start_time:.2f} sec")
    print(f"📏 Speech Rate: {wpm:.2f} WPM")

    if wpm > 160:
        print("🔴 Speaking too fast!")
    elif wpm < 90:
        print("🟡 Speaking too slow.")
    else:
        print("🟢 Speaking rate is normal.")