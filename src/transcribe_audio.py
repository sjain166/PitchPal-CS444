import torch
import librosa
import numpy as np
import json
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf

# ✅ Define the Hugging Face Model ID
model_id = "nyrahealth/CrisperWhisper"

# ✅ Select Device (GPU or CPU)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ✅ Load CrisperWhisper Model
print("🔄 Loading CrisperWhisper model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)

# ✅ Load the Processor (Tokenization + Feature Extraction)
processor = AutoProcessor.from_pretrained(model_id)

# ✅ Define Audio Path
audio_path = "../data/Pitch-Sample/audio_wav_1_35.wav"

# ✅ Load and Convert Audio to 16kHz
print("🔄 Loading and processing audio...")
audio, sr = librosa.load(audio_path, sr=16000)  # Convert to 16kHz sample rate

# ✅ Define Chunking Parameters
chunk_length_s = 30  # 🔥 Max length Whisper can handle at once
stride_length_s = 0.2   # 🔥 Use a larger stride to avoid missing words

# ✅ Split Audio into Chunks
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

# ✅ Initialize Speech-to-Text Pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    return_timestamps="word",  # ✅ Get timestamps for each word
    device=device,
)

# ✅ Transcribe Each Chunk and Store Results with Correct Timestamps
transcribed_text = ""
word_timestamps = {}  # Dictionary to store timestamps per word

for i, (chunk, chunk_start_time) in enumerate(zip(chunks, chunk_start_times)):
    print(f"🔄 Transcribing chunk {i+1}/{len(chunks)}...")
    chunk_audio_path = f"temp_chunk_{i}.wav"
    sf.write(chunk_audio_path, chunk, sr)  # Save temp chunk

    hf_pipeline_output = pipe(chunk_audio_path)  # Run transcription

    # ✅ Check if output contains timestamps
    if "chunks" in hf_pipeline_output:
        for word_obj in hf_pipeline_output["chunks"]:
            word = word_obj["text"]
            word_start = word_obj["timestamp"][0]  # Get start timestamp
            word_end = word_obj["timestamp"][1]  # Get end timestamp

            # ✅ Handle missing timestamps (NoneType issue)
            if word_start is None or word_end is None:
                print(f"⚠️ Skipping word '{word}' due to missing timestamps.")
                continue  # Skip this word

            # ✅ FIX: Correct timestamp adjustment using **chunk_start_time**
            adjusted_word_start = chunk_start_time + word_start
            adjusted_word_end = chunk_start_time + word_end

            # ✅ Debugging print to verify timestamps
            # print(f"✅ {word}: (Start: {adjusted_word_start:.2f}s, End: {adjusted_word_end:.2f}s)")

            # word_timestamps[word] = {"start": round(adjusted_word_start, 2), "end": round(adjusted_word_end, 2)}
            word_timestamps.setdefault(word, []).append({"start": adjusted_word_start, "end": adjusted_word_end})

    else:
        print(f"⚠️ Warning: No timestamps found in chunk {i+1}")

    transcribed_text += hf_pipeline_output["text"] + " "  # Merge results

# ✅ Save Full Transcription to File
transcription_path = "../data/Pitch-Sample/sample01_transcription.txt"
with open(transcription_path, "w") as f:
    f.write(transcribed_text.strip())

# ✅ Save Word Timestamps for Further Analysis
timestamp_path = "../data/Pitch-Sample/sample01_timestamps.json"
with open(timestamp_path, "w") as f:
    json.dump(word_timestamps, f, indent=4)

# ✅ Print Final Output
print("✅ Transcription complete! Saved to:", transcription_path)
print("🔍 Timestamps saved to:", timestamp_path)
print("Transcribed Text:", transcribed_text.strip())