# # scripts/wrap_opengaze.py

# """
# Wraps OpenGaze to:
# 1. Run eye contact detection on a given input video.
# 2. Generate an annotated output video with bounding boxes and labels.
# 3. Produce an eye contact timeline CSV.

# Requirements:
# - OpenGaze repo cloned and `run.py` supports --eye-contact-threshold and --output-json

# Usage:
#     python scripts/wrap_opengaze.py \
#         --input-video data/Pitch-Sample/sample.mp4 \
#         --output-video data/output_annotated.mp4 \
#         --timeline-csv data/eye_contact_timeline.csv
# """

# import argparse
# import subprocess
# import json
# import os
# import cv2
# import pandas as pd
# from pathlib import Path


# def run_opengaze(input_video: str, json_output: str, threshold=0.3):
#     cmd = [
#         'python', 'OpenGaze/run.py',
#         '--video', input_video,
#         '--eye-contact-threshold', str(threshold),
#         '--output-json', json_output
#     ]
#     print("[INFO] Running OpenGaze...")
#     subprocess.run(cmd, check=True)


# def annotate_video(input_video: str, json_path: str, output_path: str, csv_path: str):
#     print("[INFO] Annotating video using OpenGaze output...")
#     with open(json_path, 'r') as f:
#         data = json.load(f)

#     cap = cv2.VideoCapture(input_video)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     timeline = []

#     for idx, frame_data in enumerate(data['frames']):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         eye_contact = frame_data.get('eye_contact', False)
#         gaze_vector = frame_data.get('gaze_vector', [0, 0, 0])
#         msg = "Eye Contact" if eye_contact else "No Eye Contact"
#         color = (0, 255, 0) if eye_contact else (0, 0, 255)

#         cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
#         out.write(frame)

#         timeline.append({
#             'frame': idx,
#             'eye_contact': int(eye_contact),
#             'gaze_vector_x': gaze_vector[0],
#             'gaze_vector_y': gaze_vector[1],
#             'gaze_vector_z': gaze_vector[2]
#         })

#     cap.release()
#     out.release()

#     df = pd.DataFrame(timeline)
#     df.to_csv(csv_path, index=False)
#     print(f"[✓] Annotated video saved: {output_path}")
#     print(f"[✓] Timeline CSV saved: {csv_path}")


# def main():
#     parser = argparse.ArgumentParser(description="Wrap OpenGaze eye contact detection")
#     parser.add_argument('--input-video', required=True)
#     parser.add_argument('--output-video', required=True)
#     parser.add_argument('--timeline-csv', required=True)
#     parser.add_argument('--threshold', type=float, default=0.3)
#     args = parser.parse_args()

#     json_tmp = 'opengaze_output.json'
#     run_opengaze(args.input_video, json_tmp, threshold=args.threshold)
#     annotate_video(args.input_video, json_tmp, args.output_video, args.timeline_csv)
#     os.remove(json_tmp)


# if __name__ == '__main__':
#     main()


import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import argparse
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "models" / "eye-contact-cnn"))
from model import model_static  # should point to your eye-contact model.py

import dlib
from colour import Color
import time

# Constants
CNN_FACE_MODEL = "models/eye-contact-cnn/data/mmod_human_face_detector.dat"

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def draw_rect(draw, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    draw.line(points, fill=outline, width=width)

def load_model(model_path, device):
    model = model_static()
    snapshot = torch.load(model_path, map_location=device)
    model.load_state_dict(snapshot)
    model.to(device)
    model.eval()
    return model

def run_inference(video_path, model_path, output_path):
    device = get_device()
    print(f"[INFO] Using device: {device}")

    start = time.time()
    # Load model
    model = load_model(model_path, device)

    # Video I/O
    cap = cv2.VideoCapture(str(video_path))
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = 24 # cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Face detector
    detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)

    # Font & colors
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    red = Color("red")
    colors = list(red.range_to(Color("green"), 10))

    # Transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_frame)

        detections = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
        for d in detections:
            rect = d.rect
            l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom()
            l = max(int(l - 0.2 * (r - l)), 0)
            r = min(int(r + 0.2 * (r - l)), frame.shape[1])
            t = max(int(t - 0.2 * (b - t)), 0)
            b = min(int(b + 0.2 * (b - t)), frame.shape[0])
            face = pil_frame.crop((l, t, r, b))

            img_tensor = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                score = torch.sigmoid(model(img_tensor)).item()

            color_idx = min(int(round(score * 10)), 9)
            draw_rect(draw, [(l, t), (r, b)], outline=colors[color_idx].hex, width=4)
            draw.text((l, b), f"{score:.2f}", fill="white", font=font)

        # Write frame
        final = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
        out.write(final)
        frame_idx += 1

    cap.release()
    out.release()
    end = time.time()
    print(f"[INFO] Inference completed in {end - start:.2f} seconds")
    print(f"[INFO] Processed {frame_idx} frames")
    print(f"[INFO] Annotated video saved to: {output_path}")

# CLI usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Eye Contact")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", required=True, help="Path to .pkl weights")
    parser.add_argument("--output", required=True, help="Path to save output video")
    args = parser.parse_args()

    run_inference(Path(args.video), Path(args.model), Path(args.output))