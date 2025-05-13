import torch
import torchaudio
import torchaudio.transforms as T
from train_emotion_model import EmotionClassifier
import joblib

# ---- SETTINGS ----
MODEL_PATH = "/Users/aryangupta/Desktop/Emotion_Training/best_emotion_model_val_4.pth"
WAV_PATH = "/Users/aryangupta/Desktop/Emotion_Training/Emotion_Speech_Dataset/0012/Sad/0012_001069.wav"
ENCODER_PATH = "label_encoder.pkl"
SAMPLE_RATE = 16000  # or whatever your dataset used

# ---- DEVICE ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- LOAD LABEL ENCODER ----
label_encoder = joblib.load(ENCODER_PATH)
LABELS = label_encoder.classes_

# ---- LOAD MODEL ----
model = EmotionClassifier(input_dim=40, hidden_dim=64, num_classes=len(LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ---- LOAD AUDIO ----
waveform, sr = torchaudio.load(WAV_PATH)
waveform = waveform.mean(dim=0, keepdim=True)  # mono

# Resample if needed
if sr != SAMPLE_RATE:
    waveform = T.Resample(sr, SAMPLE_RATE)(waveform)

# Trim silence (VAD)
waveform = T.Vad(sample_rate=SAMPLE_RATE)(waveform)

# Optional: Normalize volume
if waveform.numel() > 0 and waveform.abs().max() > 0:
    waveform = waveform / waveform.abs().max()
else:
    raise ValueError("⚠️ Audio is silent or empty after VAD trimming. Try another recording.")

# ---- MFCC ----
transform = T.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=40, melkwargs={"n_mels": 64})
mfcc = transform(waveform).squeeze(0).T  # [Time, Features]

# ---- INFERENCE ----
with torch.no_grad():
    input_tensor = mfcc.unsqueeze(0).to(device)
    output = model(input_tensor)
    print(output)
    predicted_idx = output.argmax(dim=1).item()
    predicted_label = LABELS[predicted_idx]

    print("Predicted index:", predicted_idx)
    print("Predicted label:", predicted_label)

print(f"🎙️ Predicted Emotion: {predicted_label}")