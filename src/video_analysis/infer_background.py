# utils/infer_background_noise.py

import argparse
import json
import cv2
import torch
import torchvision
from torchvision.transforms import functional as F

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

PERSON_IDX = 1


def load_detector():
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    model.eval()
    return model

def get_device(preferred):
    if preferred == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    return torch.device(preferred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-video', required=True)
    parser.add_argument('--output-json', required=False, default="output/background_noise.json")
    parser.add_argument('--threshold', type=int, default=1, help='Max objects/persons before flagging')
    parser.add_argument('--window-sec', type=int, default=2)
    args = parser.parse_args()

    device = get_device('auto')
    model = load_detector().to(device)

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    interval_len = int(fps * args.window_sec)
    frame_idx = 0
    frame_flags = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(tensor)[0]

        num_persons = sum(p.item() > 0.6 and label == PERSON_IDX for p, label in zip(preds['scores'], preds['labels']))
        num_objects = sum(p.item() > 0.6 and label != PERSON_IDX for p, label in zip(preds['scores'], preds['labels']))

        flag = (num_persons > 1 or num_objects > args.threshold)
        frame_flags.append((frame_idx, flag))
        frame_idx += 1

    cap.release()

    # Group flagged intervals
    intervals = []
    start_idx = 0
    while start_idx < len(frame_flags):
        end_idx = min(start_idx + interval_len, len(frame_flags))
        window = frame_flags[start_idx:end_idx]
        num_flags = sum(1 for _, flagged in window if flagged)
        if num_flags >= len(window) // 2:
            start_time = window[0][0] / fps
            end_time = window[-1][0] / fps
            intervals.append({
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2)
            })
        start_idx += interval_len

    with open(args.output_json, 'w') as f:
        json.dump(intervals, f, indent=2)
    print(f"Saved background noise intervals to {args.output_json}")


if __name__ == '__main__':
    main()

# import argparse
# import json
# import cv2
# import torch
# from torchvision.transforms import functional as F
# from torchvision.models.detection import ssdlite320_mobilenet_v3_large

# PERSON_IDX = 1

# def get_device(preferred):
#     if preferred == 'auto':
#         if torch.backends.mps.is_available():
#             return torch.device('mps')
#         elif torch.cuda.is_available():
#             return torch.device('cuda')
#         else:
#             return torch.device('cpu')
#     return torch.device(preferred)

# def load_model(device):
#     model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
#     model.eval()
#     model.to(device)
#     return model

# def detect_noisy_intervals(frame_scores, fps, window_seconds=2, threshold=1):
#     interval_length = int(window_seconds * fps)
#     noisy_windows = []
#     scores = [s for _, s in frame_scores]
#     times = [f / fps for f, _ in frame_scores]

#     start_idx = 0
#     while start_idx < len(scores):
#         last_idx = min(start_idx + interval_length, len(scores))
#         window_scores = scores[start_idx:last_idx]
#         count_above_thresh = sum(s > threshold for s in window_scores)

#         if count_above_thresh >= 0.5 * len(window_scores):
#             start_time = times[start_idx]
#             end_time = times[last_idx - 1]
#             if noisy_windows and start_time <= noisy_windows[-1][1]:
#                 noisy_windows[-1][1] = end_time
#             else:
#                 noisy_windows.append([start_time, end_time])

#         start_idx += interval_length

#     return noisy_windows

# def main():
#     parser = argparse.ArgumentParser(description="Infer background noise events in video")
#     parser.add_argument('--input-video', required=True, help='Path to input MP4')
#     parser.add_argument('--output-json', required=True, help='Where to save JSON output')
#     parser.add_argument('--threshold', type=int, default=1, help='Object/person count threshold')
#     parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='Device')
#     parser.add_argument('--window-sec', type=int, default=5)
#     args = parser.parse_args()

#     device = get_device(args.device)
#     model = load_model(device)

#     cap = cv2.VideoCapture(args.input_video)
#     if not cap.isOpened():
#         raise IOError(f"Cannot open video: {args.input_video}")

#     fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
#     frame_idx = 0
#     frame_scores = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         tensor = F.to_tensor(img).unsqueeze(0).to(device)

#         with torch.no_grad():
#             preds = model(tensor)[0]

#         num_persons = sum(score.item() > 0.6 and label.item() == PERSON_IDX
#                           for score, label in zip(preds['scores'], preds['labels']))
#         num_objects = sum(score.item() > 0.6 and label.item() != PERSON_IDX
#                           for score, label in zip(preds['scores'], preds['labels']))

#         score = num_persons + num_objects
#         frame_scores.append((frame_idx, score))
#         frame_idx += 1

#     cap.release()

#     intervals = detect_noisy_intervals(frame_scores, fps, window_seconds=args.window_sec, threshold=args.threshold)
#     output = [{'start_time': round(start, 2), 'end_time': round(end, 2)} for start, end in intervals]

#     with open(args.output_json, 'w') as f:
#         json.dump(output, f, indent=2)
#     print(f"Saved background noise intervals to {args.output_json}")

# if __name__ == '__main__':
#     main()


