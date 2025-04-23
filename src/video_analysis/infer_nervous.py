import argparse
import csv
import json
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
from collections import deque


def get_device(preferred):
    if preferred == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    return torch.device(preferred)


def load_model(model_path, device):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def detect_nervous_intervals(frame_scores, fps, window_seconds=5, threshold=0.5):
    # print(f"Frame scores: {frame_scores}")
    for frame_idx, score in frame_scores:
        if score >= threshold:
            print(f"Frame {frame_idx}: Nervousness detected with score {score}")
        else:
            print(f"Frame {frame_idx}: No nervousness detected with score {score}")

    interval_length = int(window_seconds * fps)
    nervous_windows = []
    scores = [s for _, s in frame_scores]
    times = [f / fps for f, _ in frame_scores]

    start_idx = 0
    while start_idx  <= len(scores):
        last_idx = min(start_idx + interval_length , len(scores))
        window_scores = scores[start_idx: last_idx]
        count_above_thresh = sum(s >= threshold for s in window_scores)

        if count_above_thresh >= 0.5 * interval_length:
            start_time = times[start_idx]
            end_time = times[last_idx - 1]
            # Merge with previous interval if overlapping
            if nervous_windows and start_time <= nervous_windows[-1][1]:
                nervous_windows[-1][1] = end_time
            else:
                nervous_windows.append([start_time, end_time])
        start_idx += interval_length  # slide window
    return nervous_windows


def main():
    """
    python src/cv-inference-api/utils/infer_nervous.py \
    --input-video data/Pitch-Sample/sample.mp4 \
    --model-path models/nervous_classifier/best_model.pth \
    --output-json data/nervous_timeline.json
    """
    parser = argparse.ArgumentParser(description="Infer nervousness events in video")
    parser.add_argument('--input-video', required=True, help='Path to input MP4')
    parser.add_argument('--model-path', required=False, help='Path to trained .pth model', default='models/nervous_classifier/best_model.pth')
    parser.add_argument('--threshold', type=float, default=0.55, help='Probability threshold')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='Device')
    parser.add_argument('--output-json', required=False, help='Where to save JSON output', default="output/nervous_timeline.json")
    args = parser.parse_args()

    device = get_device(args.device)
    model = load_model(args.model_path, device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(48),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_idx = 0
    frame_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)
        score = 0.0

        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size != 0:
                inp = transform(face_roi).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(inp)
                    probs = torch.softmax(logits, dim=1)
                    score = probs[0, 0].item()  # nervous class

        frame_scores.append((frame_idx, score))
        frame_idx += 1

    cap.release()

    intervals = detect_nervous_intervals(frame_scores, fps, window_seconds=5, threshold=args.threshold)
    output = [{'start_time': round(start, 2), 'end_time': round(end, 2)} for start, end in intervals]

    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved nervous intervals to {args.output_json}")


if __name__ == '__main__':
    main()


