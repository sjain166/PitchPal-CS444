import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import numpy as np
import json
import dlib
from pathlib import Path
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "models" / "eye-contact-cnn"))
from model import model_static

# Constants
CNN_FACE_MODEL = "models/eye-contact-cnn/data/mmod_human_face_detector.dat"



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
    model = model_static()
    snapshot = torch.load(model_path, map_location=device)
    model.load_state_dict(snapshot)
    model.to(device)
    model.eval()
    return model

def detect_eye_contact_intervals(scores, fps, window_seconds=5, threshold=0.5):
    interval_len = int(window_seconds * fps)
    intervals = []
    start_idx = 0
    times = [f / fps for f, _ in scores]
    values = [s for _, s in scores]

    while start_idx < len(values):
        end_idx = min(start_idx + interval_len, len(values))
        window = values[start_idx:end_idx]
        no_eye_contact = sum(1 for v in window if v < threshold)

        if no_eye_contact >= 0.5 * len(window):
            start_time = times[start_idx]
            end_time = times[end_idx - 1]
            if intervals and start_time <= intervals[-1][1]:
                intervals[-1][1] = end_time
            else:
                intervals.append([start_time, end_time])
        start_idx += interval_len

    return intervals

def main():
    """
    Usage:
        python src/video_analysis/infer_eye.py \
        --video data/Pitch-Sample/aryan_short.mp4 \
        --model models/eye-contact-cnn/data/model_weights.pkl \
        --output-json data/eye_contact_timeline.json
    """
    parser = argparse.ArgumentParser(description="Infer Eye Contact Events")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", required=False, help="Path to .pkl weights", default="models/eye-contact-cnn/data/model_weights.pkl")
    parser.add_argument("--output-json", required=False, help="Path to save output JSON" , default="src/tests/results/eye_discontact_timeline.json")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score below which we consider no eye contact")
    args = parser.parse_args()

    device = get_device("auto")
    model = load_model(args.model, device)

    detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_idx = 0
    scores = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
        score = 1.0  # assume eye contact unless proven otherwise

        if detections:
            rect = detections[0].rect
            l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom()
            l = max(int(l - 0.2 * (r - l)), 0)
            r = min(int(r + 0.2 * (r - l)), frame.shape[1])
            t = max(int(t - 0.2 * (b - t)), 0)
            b = min(int(b + 0.2 * (b - t)), frame.shape[0])

            face = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).crop((l, t, r, b))
            img_tensor = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                score = torch.sigmoid(model(img_tensor)).item()

        scores.append((frame_idx, score))
        frame_idx += 1

    cap.release()

    intervals = detect_eye_contact_intervals(scores, fps, window_seconds=5, threshold=args.threshold)
    output = [{"start_time": round(s, 2), "end_time": round(e, 2)} for s, e in intervals]

    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[✓] Saved no-eye-contact intervals to {args.output_json}")

if __name__ == '__main__':
    main()
