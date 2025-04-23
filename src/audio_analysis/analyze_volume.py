import librosa
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import librosa.display
import argparse

# Import necessary libraries
# --- CLI argument parsing ---
parser = argparse.ArgumentParser(description="Volume interval analysis using RMS energy")
parser.add_argument("audio_path", help="Path to the input .wav audio file")
parser.add_argument("--json_output_path", default="src/tests/results/volume_report.json", help="Path to store the JSON result file")
parser.add_argument("--plot_output_path", default="src/tests/results/volume_plot.png", help="Path to store the volume plot image (PNG)")
args = parser.parse_args()

# --- Load audio file ---
# y = audio waveform, sr = sampling rate
y, sr = librosa.load(args.audio_path, sr=None)

# Compute root mean square (RMS) energy for each frame
rms = librosa.feature.rms(y=y)[0]

# Get timestamps for each RMS frame
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

# --- Analysis Parameters ---
WINDOW_SIZE = 2.0  # analysis window in seconds
STEP_SIZE = 1.0    # step size for moving window in seconds
SILENCE_THRESHOLD = 0.005  # RMS below this is treated as silence
MIN_DURATION = 2.0  # minimum duration (in seconds) to consider an interval significant

# --- Compute dynamic thresholds based on non-silent parts ---
non_silent_rms = rms[rms > SILENCE_THRESHOLD]
NORMAL_MIN = np.percentile(non_silent_rms, 25)  # low energy bound
NORMAL_MAX = np.percentile(non_silent_rms, 75)  # high energy bound

# Container for loud and inaudible intervals
intervals = {
    "loud": [],
    "inaudible": []
}

# --- Analyze intervals using sliding window ---
def analyze_intervals(rms, times, min_thres, max_thres):
    start = 0
    end = WINDOW_SIZE
    while end <= times[-1]:
        start_idx = np.searchsorted(times, start)
        end_idx = np.searchsorted(times, end)

        if end_idx - start_idx < 2:
            # Skip if window is too narrow
            start += STEP_SIZE
            end += STEP_SIZE
            continue

        rms_seg = rms[start_idx:end_idx]
        time_seg = times[start_idx:end_idx]
        avg_rms = np.mean(rms_seg)

        # Ignore windows that are completely silent
        if avg_rms < SILENCE_THRESHOLD:
            start += STEP_SIZE
            end += STEP_SIZE
            continue

        # Classify based on average RMS
        if avg_rms < min_thres:
            intervals["inaudible"].append({"start_time": round(start, 2), "end_time": round(end, 2)})
        elif avg_rms > max_thres:
            intervals["loud"].append({"start_time": round(start, 2), "end_time": round(end, 2)})

        # Slide the window
        start += STEP_SIZE
        end += STEP_SIZE

# Run volume classification
analyze_intervals(rms, times, NORMAL_MIN, NORMAL_MAX)

# --- Merge overlapping intervals to reduce redundancy ---
def merge_chunks(intervals):
    if not intervals:
        return []
    merged = [intervals[0]]
    for curr in intervals[1:]:
        prev = merged[-1]
        if curr["start_time"] <= prev["end_time"]:
            # Extend overlapping intervals
            prev["end_time"] = max(prev["end_time"], curr["end_time"])
        else:
            merged.append(curr)
    # Filter out intervals shorter than MIN_DURATION
    return [i for i in merged if i["end_time"] - i["start_time"] >= MIN_DURATION]

# Apply merging to both categories
intervals["inaudible"] = merge_chunks(sorted(intervals["inaudible"], key=lambda x: x["start_time"]))
intervals["loud"] = merge_chunks(sorted(intervals["loud"], key=lambda x: x["start_time"]))

# --- Save JSON output ---
os.makedirs(os.path.dirname(args.json_output_path), exist_ok=True)
with open(args.json_output_path, "w") as f:
    json.dump(intervals, f, indent=4)

# --- Generate and save waveform + RMS plot ---
os.makedirs(os.path.dirname(args.plot_output_path), exist_ok=True)
plt.figure(figsize=(12, 4))

# Plot waveform and RMS curve
librosa.display.waveshow(y, sr=sr, alpha=0.5, label="Waveform")
plt.plot(times, rms, label="RMS Energy", color='black')

# Add threshold lines
plt.axhline(NORMAL_MAX, color='green', linestyle='--', label=f"Loud Thresh ({NORMAL_MAX:.3f})")
plt.axhline(NORMAL_MIN, color='red', linestyle='--', label=f"Inaudible Thresh ({NORMAL_MIN:.3f})")

# Label plot
plt.xlabel("Time (s)")
plt.ylabel("RMS Energy")
plt.title("Volume Analysis Based on RMS Energy")
plt.legend()
plt.tight_layout()

# Save and close the figure
plt.savefig(args.plot_output_path)
plt.close()

# Final confirmation message
print(f"✅ Volume Analysis Report saved to: {args.json_output_path}")