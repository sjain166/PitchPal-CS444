import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
import torchaudio.transforms as T
import torch.nn.utils.rnn as rnn_utils
from collections import Counter
import joblib
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Optimize backend
torch.backends.cudnn.benchmark = True

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_sequences = rnn_utils.pad_sequence(sequences, batch_first=True)
    return padded_sequences, torch.tensor(labels)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()

class EmotionDataset(Dataset):
    def __init__(self, root_dir, data_list, label_encoder):
        self.data = data_list
        self.label_encoder = label_encoder
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav_path, transcript, label_str = self.data[idx]
        label = self.label_encoder.transform([label_str])[0]

        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform.mean(dim=0, keepdim=True)
        transform = T.MFCC(sample_rate=sr, n_mfcc=40, melkwargs={"n_mels": 64})
        mfcc = transform(waveform).squeeze(0)

        return mfcc.T, label

def load_dataset(root_dir):
    data = []
    all_labels = []
    for session in os.listdir(root_dir):
        session_path = os.path.join(root_dir, session)
        if not os.path.isdir(session_path):
            continue
        txt_path = os.path.join(session_path, f"{session}.txt")
        if not os.path.exists(txt_path):
            continue

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                file_id, transcript, label = parts
                wav_file = os.path.join(session_path, label, f"{file_id}.wav")
                if os.path.exists(wav_file):
                    data.append((wav_file, transcript, label))
                    all_labels.append(label)

    print("[INFO] Label distribution:", Counter(all_labels))

    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    joblib.dump(label_encoder, "label_encoder.pkl")

    return data, label_encoder

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=64, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

def train():
    print(f"Using device: {device}")
    data, label_encoder = load_dataset("/Users/aryangupta/Desktop/Emotion_Training/Emotion_Speech_Dataset")

    train_data, val_data = train_test_split(data, test_size=0.2, stratify=[d[2] for d in data], random_state=42)

    train_dataset = EmotionDataset(root_dir=None, data_list=train_data, label_encoder=label_encoder)
    val_dataset = EmotionDataset(root_dir=None, data_list=val_data, label_encoder=label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=use_amp)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=use_amp)

    model = EmotionClassifier(num_classes=len(label_encoder.classes_)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(0, 100):
        model.train()
        total_loss = 0

        for mfccs, labels in tqdm(train_loader):
            mfccs = mfccs.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                outputs = model(mfccs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for mfccs, labels in val_loader:
                mfccs = mfccs.to(device)
                labels = labels.to(device)
                outputs = model(mfccs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_emotion_model_val_4.pth")
            print(f"✅ Saved best model (val loss {best_val_loss:.4f})")

        if (epoch + 1) % 10 == 0:
            print("[DEBUG] Classification report (validation):")
            print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    print("🏁 Training complete.")
    
    # Plot the loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()

if __name__ == "__main__":
    train()