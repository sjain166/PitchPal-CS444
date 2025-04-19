import torch
import librosa
import numpy as np
import json
import sys
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf

#  Define Audio Path
audio_path = sys.argv[1]
timestamp_path = "./tests/timestamp.json"
transcription_path = "./tests/transcription.txt"

# Define the Hugging Face Model ID
model_id = "nyrahealth/CrisperWhisper"

# Select Device (GPU or CPU)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load CrisperWhisper Model
print("🔄 Loading CrisperWhisper model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)

# Load the Processor (Tokenization + Feature Extraction)
processor = AutoProcessor.from_pretrained(model_id)

# Load and Convert Audio to 16kHz
print("🔄 Loading and processing audio...")
audio, sr = librosa.load(audio_path, sr=16000)  #Convert to 16kHz sample rate

# Define Chunking Parameters
chunk_length_s = 30  # 🔥 Max length Whisper can handle at once
stride_length_s = 0.2   # 🔥 0.2s overlap ensures smooth transitions

# Split Audio into Chunks
def split_audio(audio, sr, chunk_length_s, stride_length_s):
    chunk_samples = int(chunk_length_s * sr)  # Convert seconds to samples
    stride_samples = int(stride_length_s * sr)

    chunks = []
    timestamps = []  # To store timestamp of each chunk
    for start in range(0, len(audio), chunk_samples - stride_samples):
        end = min(start + chunk_samples, len(audio))
        chunks.append(audio[start:end])
        timestamps.append(start / sr)  # Convert sample index to time in seconds

    return chunks, timestamps

chunks, chunk_start_times = split_audio(audio, sr, chunk_length_s, stride_length_s)

# Initialize Speech-to-Text Pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    return_timestamps="word",  #  Get timestamps for each word
    device=device,
)

# Transcribe Each Chunk and Store Results with Correct Timestamps
transcribed_text = ""
word_timestamps = []  # Dictionary to store timestamps per word
temp_chunk_files = []  # List to keep track of temporary chunk files

for i, (chunk, chunk_start_time) in enumerate(zip(chunks, chunk_start_times)):
    print(f"🔄 Transcribing chunk {i+1}/{len(chunks)}...")
    chunk_audio_path = f"./tests/temp_chunk_{i}.wav"
    sf.write(chunk_audio_path, chunk, sr)  # Save temp chunk
    temp_chunk_files.append(chunk_audio_path)
    hf_pipeline_output = pipe(chunk_audio_path)  # Run transcription

    #  Check if output contains timestamps
    if "chunks" in hf_pipeline_output:
        for word_obj in hf_pipeline_output["chunks"]:
            word = word_obj["text"]
            word_start = word_obj["timestamp"][0]  # Get start timestamp
            word_end = word_obj["timestamp"][1]  # Get end timestamp

            #  Handle missing timestamps (NoneType issue)
            if word_start is None or word_end is None:
                print(f"⚠️ Skipping word '{word}' due to missing timestamps.")
                continue  # Skip this word

            adjusted_word_start = chunk_start_time + word_start
            adjusted_word_end = chunk_start_time + word_end
            word_timestamps.append({
                "word": word,
                "start_time": adjusted_word_start,
                "end_time": adjusted_word_end
            })

    else:
        print(f"⚠️ Warning: No timestamps found in chunk {i+1}")

    transcribed_text += hf_pipeline_output["text"] + " "  # Merge results


# Save Full Transcription to File
with open(transcription_path, "w") as f:
    f.write(transcribed_text.strip())

# Save Word Timestamps for Further Analysis
with open(timestamp_path, "w") as f:
    json.dump(word_timestamps, f, indent=4)
    
for file_path in temp_chunk_files:
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"⚠️ Error deleting file {file_path}: {e}")

# Print Final Output
print("✅ Transcription saved to: ", transcription_path)
print("✅ Timestamps saved to:", timestamp_path)