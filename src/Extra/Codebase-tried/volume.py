import torchaudio
import torch
import numpy as np

# Load your audio
waveform, sample_rate = torchaudio.load("/Users/aryangupta/Desktop/Emotion_Training/Project/loud.wav")  # Replace with your file

# Convert stereo to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Compute RMS (Root Mean Square)
rms = torch.sqrt(torch.mean(waveform**2))

# Avoid log(0)
if rms.item() == 0:
    dbfs = -float('inf')
else:
    dbfs = 20 * torch.log10(rms / 1.0)  # max amplitude = 1.0 for float audio

# Interpretation function
def interpret_loudness(dbfs_value):
    if dbfs_value > -30:
        return "🔴 Too Loud"
    elif dbfs_value < -40:
        return "🟡 Too Soft"
    else:
        return "🟢 Normal Volume"

# Output
print(f"🔊 dBFS: {dbfs.item():.2f}")
print(f"📝 Loudness Interpretation: {interpret_loudness(dbfs.item())}")