import torch
import librosa
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

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
audio_path = "../data/Pitch-Sample/personal_filler.wav"

# ✅ Load and Convert Audio to 16kHz
print("🔄 Loading and processing audio...")
audio, sr = librosa.load(audio_path, sr=16000)  # Convert to 16kHz sample rate

# ✅ Define Chunking Parameters
chunk_length_s = 30  # 🔥 Max length Whisper can handle at once
stride_length_s = 5   # 🔥 5s overlap ensures smooth transitions

# ✅ Split Audio into Chunks
def split_audio(audio, sr, chunk_length_s, stride_length_s):
    chunk_samples = int(chunk_length_s * sr)  # Convert seconds to samples
    stride_samples = int(stride_length_s * sr)
    
    chunks = []
    for start in range(0, len(audio), chunk_samples - stride_samples):
        end = min(start + chunk_samples, len(audio))
        chunks.append(audio[start:end])
    
    return chunks

chunks = split_audio(audio, sr, chunk_length_s, stride_length_s)

# ✅ Initialize Speech-to-Text Pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# ✅ Transcribe Each Chunk and Merge Results
transcribed_text = ""
for i, chunk in enumerate(chunks):
    print(f"🔄 Transcribing chunk {i+1}/{len(chunks)}...")
    chunk_audio_path = f"temp_chunk_{i}.wav"
    librosa.output.write_wav(chunk_audio_path, chunk, sr)  # Save temp chunk
    
    hf_pipeline_output = pipe(chunk_audio_path)  # Run transcription
    transcribed_text += hf_pipeline_output["text"] + " "  # Merge results

# ✅ Save Full Transcription to File
transcription_path = "../data/Pitch-Sample/sample01_transcription.txt"
with open(transcription_path, "w") as f:
    f.write(transcribed_text.strip())

# ✅ Print Final Output
print("✅ Transcription complete! Saved to:", transcription_path)
print("Transcribed Text:", transcribed_text.strip())