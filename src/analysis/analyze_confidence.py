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

y, sr = librosa.load(args.audio_path, sr=None)

# Parameters
frame_length = 2048
hop_length = 512
SILENCE_THRESHOLD = 0.005
WINDOW_SIZE = 2.0
STEP_SIZE = 1.0

# ✅ Initialize intervals
intervals = {
    "underconfident": [],
    "overconfident": []
}

# ✅ Compute RMS
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

# ✅ Dynamic thresholds
NORMAL_MIN = np.percentile(rms, 25)
NORMAL_MAX = np.percentile(rms, 75)

# ✅ Sliding window logic
def compute_area_between_bounds(rms_segment, time_segment, lower, upper):
    mask = (rms_segment >= lower) & (rms_segment <= upper)
    return np.trapz(rms_segment[mask], time_segment[mask])

start = 0
end = WINDOW_SIZE

while end <= times[-1]:
    start_idx = np.searchsorted(times, start)
    end_idx = np.searchsorted(times, end)
    rms_seg = rms[start_idx:end_idx]
    time_seg = times[start_idx:end_idx]

    if len(rms_seg) == 0 or np.mean(rms_seg) < SILENCE_THRESHOLD:
        start += STEP_SIZE
        end += STEP_SIZE
        continue

    total_area = np.trapz(rms_seg, time_seg)
    if total_area == 0:
        start += STEP_SIZE
        end += STEP_SIZE
        continue

    over_area = compute_area_between_bounds(rms_seg, time_seg, NORMAL_MAX, np.inf)
    under_area = compute_area_between_bounds(rms_seg, time_seg, -np.inf, NORMAL_MIN)

    over_ratio = over_area / total_area
    under_ratio = under_area / total_area

    if over_ratio > 0.7 and over_area > 0.01:
        intervals["overconfident"].append({"start": round(start, 2), "end": round(end, 2)})
    elif under_ratio > 0.7 and under_area > 0.01:
        intervals["underconfident"].append({"start": round(start, 2), "end": round(end, 2)})

    start += STEP_SIZE
    end += STEP_SIZE

# Merge and filter intervals
def merge_intervals(intervals, gap_threshold=0.5, max_merge_gap=2.0):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x["start"])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current["start"] - last["end"] <= gap_threshold and current["start"] - last["end"] <= max_merge_gap:
            last["end"] = max(last["end"], current["end"])
        else:
            merged.append(current)
    return [{"start": round(i["start"], 2), "end": round(i["end"], 2)} for i in merged]

def filter_silent_intervals(interval_list):
    return [i for i in interval_list if i["start"] > 0.5]

def filter_significant_intervals(intervals, kind="overconfident", min_duration=1.0, std_threshold=0.7):
    filtered = []
    global_mean = np.mean(rms)
    global_std = np.std(rms)
    for interval in intervals:
        start_idx = np.searchsorted(times, interval["start"])
        end_idx = np.searchsorted(times, interval["end"])
        rms_segment = rms[start_idx:end_idx]
        duration = interval["end"] - interval["start"]

        if len(rms_segment) == 0 or duration < min_duration:
            continue

        mean_rms = np.mean(rms_segment)

        if kind == "overconfident" and mean_rms > global_mean + std_threshold * global_std:
            filtered.append(interval)
        elif kind == "underconfident" and mean_rms < global_mean - std_threshold * global_std:
            filtered.append(interval)

    return filtered

intervals["underconfident"] = filter_significant_intervals(
    merge_intervals(filter_silent_intervals(intervals["underconfident"]), max_merge_gap=1.0),
    kind="underconfident", min_duration=1.0, std_threshold=0.1
)

intervals["overconfident"] = filter_significant_intervals(
    merge_intervals(filter_silent_intervals(intervals["overconfident"]), max_merge_gap=1.0),
    kind="overconfident", min_duration=1.0, std_threshold=0.1
)

# ✅ Save JSON output
os.makedirs(os.path.dirname(args.json_output_path), exist_ok=True)
with open(args.json_output_path, "w") as f:
    json.dump(intervals, f, indent=4)

# ✅ Save plot
os.makedirs(os.path.dirname(args.plot_output_path), exist_ok=True)
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.5, label="Waveform")
plt.plot(times, rms, label="RMS Energy", color='black')
plt.axhline(NORMAL_MAX, color='green', linestyle='--', label=f"Dynamic Max ({NORMAL_MAX:.3f})")
plt.axhline(NORMAL_MIN, color='red', linestyle='--', label=f"Dynamic Min ({NORMAL_MIN:.3f})")
plt.xlabel("Time (s)")
plt.ylabel("RMS Energy")
plt.title("Confidence Analysis Based on RMS Energy")
plt.legend()
plt.tight_layout()
plt.savefig(args.plot_output_path)
plt.close()