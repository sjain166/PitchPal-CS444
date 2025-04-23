# scripts/detect_nervous_video.py

"""
Run nervousness detection on a video and annotate frames with bounding boxes when nervousness is detected.
Also generates a heatmap of nervousness frequency over time.

Usage:
    python src/video_analysis/analyze_nervous.py \
        --input-video data/Pitch-Sample/sample_5.mp4  \
        --model-path models/nervous_classifier/best_model.pth \
        --output-video data/output.mp4 \
        --threshold 0.65
"""
import argparse
from pathlib import Path
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Device selection

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


def generate_heatmap(log_file, output_path, fps, threshold):
    df = pd.read_csv(log_file)
    df['time_sec'] = df['frame'] / fps

    # Use 10-second interval bins
    df['time_bucket'] = (df['time_sec'] // 10).astype(int) * 10
    grouped = df.groupby('time_bucket')['nervous_score'].mean()
    grouped[grouped < threshold] = np.nan  # Mask values below threshold

    # normalized = (grouped - np.nanmin(grouped)) / (np.nanmax(grouped) - np.nanmin(grouped) + 1e-8)
    if grouped.isna().all():
        print("No intervals passed the nervousness threshold. Heatmap will be blank.")
        normalized = np.full_like(grouped, np.nan)
    else:
        normalized = (grouped - np.nanmin(grouped)) / (np.nanmax(grouped) - np.nanmin(grouped) + 1e-8)
        
    plt.figure(figsize=(max(12, len(grouped) // 3), 2))
    sns.heatmap([normalized], cmap='Reds', cbar=True, xticklabels=grouped.index.astype(int), vmin=0, vmax=1)
    plt.yticks([])
    plt.xticks(rotation=45)
    plt.title("Normalized Nervousness Over Time (per 10s segment)")
    plt.xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Heatmap saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect nervousness in video frames")
    parser.add_argument('--input-video',  required=True, help='Path to input MP4')
    parser.add_argument('--model-path',   required=True, help='Path to trained .pth model')
    parser.add_argument('--output-video', required=True, help='Path to save annotated MP4')
    parser.add_argument('--threshold',    type=float, default=0.5,
                        help='Probability threshold for nervous class')
    parser.add_argument('--device',       default='auto', choices=['auto','cpu','cuda','mps'],
                        help='Compute device')
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    model = load_model(args.model_path, device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(48),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.input_video}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    # CSV logging setup
    log_file = args.output_video.replace(".mp4", "_nervous_log.csv")
    log_writer = open(log_file, "w", newline="")
    csv_writer = csv.writer(log_writer)
    csv_writer.writerow(["frame", "nervous_score"])

    frame_idx = 0
    running_score = 0
    nervous_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)
        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            x1 = int(bbox.xmin * width)
            y1 = int(bbox.ymin * height)
            w_box = int(bbox.width * width)
            h_box = int(bbox.height * height)
            x2, y2 = x1 + w_box, y1 + h_box
            x1, y1 = max(0,x1), max(0,y1)
            x2, y2 = min(width,x2), min(height,y2)

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size != 0:
                inp = transform(face_roi).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(inp)
                    probs = torch.softmax(logits, dim=1)
                    score = probs[0,0].item()
                csv_writer.writerow([frame_idx, score])
                if score >= args.threshold:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    running_score += score
                    nervous_frames += 1
                    cv2.putText(frame, f"Nervous:{score:.2f}", (x1, max(0,y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    log_writer.close()
    print(f"Annotated video saved to {args.output_video}")
    print(f"Processed {frame_idx} frames.")
    print(f"Detected nervous frames: {nervous_frames}")
    if nervous_frames > 0:
        print(f"Average nervousness score: {running_score / nervous_frames:.2f}")

    heatmap_output = args.output_video.replace(".mp4", "_heatmap.png")
    generate_heatmap(log_file, heatmap_output, fps, args.threshold)


if __name__ == '__main__':
    main()