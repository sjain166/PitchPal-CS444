
import whisper

audio_path = "../data/Pitch-Sample/filler_sample.wav"
model = whisper.load_model("base")

print("🔄 Transcribing with CrisperWhisper...")
result = model.transcribe(audio_path)

transcription_path = "../data/Pitch-Sample/sample01_transcription.txt"
with open(transcription_path, "w") as f:
    f.write(result["text"])

print("✅ Transcription complete! Saved to:", transcription_path)
print("Transcribed Text:", result["text"])