import librosa
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import librosa.display
import argparse

# CLI Arguments
parser = argparse.ArgumentParser(description="Confidence interval analysis using RMS energy")
parser.add_argument("audio_path", help="Path to the input .wav audio file")
parser.add_argument("json_output_path", help="Path to store the JSON result file")
parser.add_argument("plot_output_path", help="Path to store the confidence plot image (PNG)")
args = parser.parse_args()

# Load audio
y, sr = librosa.load(args.audio_path, sr=None)
rms = librosa.feature.rms(y=y)[0]
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

# Parameters
WINDOW_SIZE = 2.0  # seconds
STEP_SIZE = 1.0  # seconds
SILENCE_THRESHOLD = 0.005  # RMS energy below this is silence
MIN_DURATION = 2.0  # seconds

# Dynamic thresholds (exclude silence from percentile)
non_silent_rms = rms[rms > SILENCE_THRESHOLD]
NORMAL_MIN = np.percentile(non_silent_rms, 25)
NORMAL_MAX = np.percentile(non_silent_rms, 75)

intervals = {
    "overconfidence": [],
    "underconfidence": []
}

# Confidence tagging using RMS window analysis
def analyze_intervals(rms, times, min_thres, max_thres):
    start = 0
    end = WINDOW_SIZE
    while end <= times[-1]:
        start_idx = np.searchsorted(times, start)
        end_idx = np.searchsorted(times, end)

        if end_idx - start_idx < 2:
            start += STEP_SIZE
            end += STEP_SIZE
            continue

        rms_seg = rms[start_idx:end_idx]
        time_seg = times[start_idx:end_idx]

        avg_rms = np.mean(rms_seg)
        if avg_rms < SILENCE_THRESHOLD:
            start += STEP_SIZE
            end += STEP_SIZE
            continue

        if avg_rms < min_thres:
            intervals["underconfidence"].append({"start_time": round(start, 2), "end_time": round(end, 2)})
        elif avg_rms > max_thres:
            intervals["overconfidence"].append({"start_time": round(start, 2), "end_time": round(end, 2)})

        start += STEP_SIZE
        end += STEP_SIZE

analyze_intervals(rms, times, NORMAL_MIN, NORMAL_MAX)

# Merge overlapping intervals
def merge_chunks(intervals):
    if not intervals:
        return []
    merged = [intervals[0]]
    for curr in intervals[1:]:
        prev = merged[-1]
        if curr["start_time"] <= prev["end_time"]:
            prev["end_time"] = max(prev["end_time"], curr["end_time"])
        else:
            merged.append(curr)
    return [i for i in merged if i["end_time"] - i["start_time"] >= MIN_DURATION]

intervals["underconfidence"] = merge_chunks(sorted(intervals["underconfidence"], key=lambda x: x["start_time"]))
intervals["overconfidence"] = merge_chunks(sorted(intervals["overconfidence"], key=lambda x: x["start_time"]))

# Save JSON result
os.makedirs(os.path.dirname(args.json_output_path), exist_ok=True)
with open(args.json_output_path, "w") as f:
    json.dump(intervals, f, indent=4)

# Save plot
os.makedirs(os.path.dirname(args.plot_output_path), exist_ok=True)
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.5, label="Waveform")
plt.plot(times, rms, label="RMS Energy", color='black')
plt.axhline(NORMAL_MAX, color='green', linestyle='--', label=f"Overconfident Thresh ({NORMAL_MAX:.3f})")
plt.axhline(NORMAL_MIN, color='red', linestyle='--', label=f"Underconfident Thresh ({NORMAL_MIN:.3f})")
plt.xlabel("Time (s)")
plt.ylabel("RMS Energy")
plt.title("Confidence Analysis Based on RMS Energy")
plt.legend()
plt.tight_layout()
plt.savefig(args.plot_output_path)
plt.close()

print(f"✅ Confidence Report saved to: {args.json_output_path}")