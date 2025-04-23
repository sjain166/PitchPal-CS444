# scripts/detect_background_noise.py

"""
Detects moving objects and multiple people in a video to verify background cleanliness:
- Warns if more than 1 person is detected in the frame
- Warns if non-static objects (cars, pets, people walking) are detected
- Draws bounding boxes on all detected objects

Usage:
    python src/video_analysis/detect_background_noise.py \
        --input-video data/Pitch-Sample/sample.mp4 \
        --output-video data/scene_noise_check.mp4
"""

import cv2
import argparse
import torch
import torchvision
from torchvision.transforms import functional as F
from pathlib import Path
from tqdm import tqdm

# COCO person class index
PERSON_IDX = 1

# COCO class labels
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
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load pretrained object detector (Faster R-CNN with ResNet50 backbone)
def load_detector():
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    model.eval()
    return model

def detect_objects(frame, model, device):
    image = F.to_tensor(frame).to(device)
    with torch.no_grad():
        predictions = model([image])[0]
    return predictions

def analyze_scene(input_path, output_path, person_threshold=0.7, obj_threshold=0.6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_detector().to(device)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    person_violation_count = 0
    motion_violation_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predictions = detect_objects(frame, model, device)

        num_people = 0

        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
            if score < obj_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label.item()] if label.item() < len(COCO_INSTANCE_CATEGORY_NAMES) else str(label.item())

            if label.item() == PERSON_IDX and score >= person_threshold:
                num_people += 1

            color = (0, 255, 0) if label.item() == PERSON_IDX else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label_name} {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if num_people > 1:
            person_violation_count += 1
            cv2.putText(frame, "[WARNING] Multiple people detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
       

        non_person_objects = [
            (label.item(), score.item())
            for label, score in zip(predictions['labels'], predictions['scores'])
            if label.item() != PERSON_IDX and score.item() >= obj_threshold
        ]

        if len(non_person_objects) > 0:
            motion_violation_count += 1
            cv2.putText(frame, "[WARNING] Moving object(s) in background", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[✓] Output saved to: {output_path}")
    print(f"Frames processed: {frame_idx}")
    print(f"Frames with multiple people: {person_violation_count}")
    print(f"Frames with background motion: {motion_violation_count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Background cleanliness analysis")
    parser.add_argument('--input-video', required=True)
    parser.add_argument('--output-video', required=True)
    args = parser.parse_args()
    analyze_scene(args.input_video, args.output_video)