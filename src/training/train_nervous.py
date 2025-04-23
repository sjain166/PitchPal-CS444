# scripts/train_nervous.py

"""
Train a binary classifier for nervousness detection using fear proxies:
- Expects data directory with `train/` and `val/`:
    train/nervous, train/not_nervous,
    val/nervous,   val/not_nervous
- Fine-tunes EfficientNet-B0 with mixed precision on A100
- Uses a weighted sampler to handle class imbalance
- Saves best model checkpoint

Usage:
    python3 src/training/train_nervous.py \
    --data-dir data/model_training_data \
    --epochs 10 \
    --batch-size 64 \
    --lr 1e-4 \
    --output-dir models/nervous_classifier
"""
import argparse
import os
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Device selection: MPS (Metal) > CUDA > CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Mixed precision setup
USE_AMP = DEVICE.type in ('cuda', 'mps')
scaler = GradScaler() if USE_AMP else None


def get_data_loaders(data_dir, batch_size):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(48, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(48),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(Path(data_dir) / 'train', transform=train_transforms)
    val_ds   = datasets.ImageFolder(Path(data_dir) / 'val',   transform=val_transforms)

    # Weighted sampler to balance classes
    counts = Counter(train_ds.targets)
    total = sum(counts.values())
    class_weights = {cls: total/count for cls, count in counts.items()}
    sample_weights = [class_weights[label] for label in train_ds.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,      num_workers=4)
    return train_loader, val_loader, train_ds.classes


def train(args):
    train_loader, val_loader, classes = get_data_loaders(args.data_dir, args.batch_size)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr * 10,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [TRAIN]', leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            if USE_AMP:
                with autocast(device_type=DEVICE.type):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct = 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch}/{args.epochs} [VAL]  ', leave=False)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels)
        # Use float32 for MPS compatibility
        epoch_acc = correct.float() / len(val_loader.dataset)
        print(f"Epoch {epoch}: Loss={epoch_loss:.4f} Val_Acc={epoch_acc:.4f}")

        # Checkpoint best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), Path(args.output_dir) / 'best_model.pth')
        scheduler.step()

    print(f"Training complete. Best Val Acc: {best_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train nervousness classifier')
    parser.add_argument('--data-dir',   required=True, help='Root dir with train/ and val/')
    parser.add_argument('--epochs',     type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr',         type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output-dir', required=True, help='Directory for checkpoints')
    args = parser.parse_args()
    print(f"Using device: {DEVICE}")
    train(args)
