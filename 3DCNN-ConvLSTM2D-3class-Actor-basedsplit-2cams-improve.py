import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import random
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time

# =======================
# Reproducibility
# =======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =======================
# Dataset preparation
# =======================
FRAMES_ROOTS = [
    "./cam2-resized_frames",
    "./cam3-resized_frames",
]
LABELS_ROOT = "./micro_activity_output"
SEQUENCE_LENGTH = 16

FALL_KEYWORDS = ["fall_"]
LYING_KEYWORDS = ["lying_", "_lying"]

with open("./f_seq_stride8.pkl", "rb") as f:
    dataset = pickle.load(f)

print(f"Loaded dataset with {len(dataset)} sequences.")

adl_count = sum(1 for _, label in dataset if label == 0)
fall_count = sum(1 for _, label in dataset if label == 1)
lying_count = sum(1 for _, label in dataset if label == 2)

print(f"Total sequences: {len(dataset)}")
print(f"ADL sequences: {adl_count}")
print(f"Fall sequences: {fall_count}")
print(f"Lie Down sequences: {lying_count}")

# =======================
# Actor-based split (deterministic)
# =======================
actor_to_samples = {}
for seq_paths, label in dataset:
    parts = seq_paths[0].split(os.sep)
    actor_folder = next(p for p in parts if p.startswith("Actor_"))
    actor_id = "_".join(actor_folder.split("_")[:2])
    actor_to_samples.setdefault(actor_id, []).append((seq_paths, label))

# Sort actors list to ensure stable order
all_actors = sorted(actor_to_samples.keys())

# Shuffle actors with fixed seed to always get same split
rng = random.Random(SEED)
rng.shuffle(all_actors)

split_idx = int(0.8 * len(all_actors))
train_actors, val_actors = all_actors[:split_idx], all_actors[split_idx:]

train_samples = [s for a in train_actors for s in actor_to_samples[a]]
val_samples = [s for a in val_actors for s in actor_to_samples[a]]

print("VAL actors:", val_actors)
print("TRAIN actors:", train_actors)
from collections import Counter
print("VAL class dist:", Counter([y for _,y in val_samples]))
print("TRAIN class dist:", Counter([y for _,y in train_samples]))


print(f"Actors: {len(all_actors)}, Train actors: {len(train_actors)}, Val actors: {len(val_actors)}")
print(f"Train sequences: {len(train_samples)}, Val sequences: {len(val_samples)}")

# =======================
# Dataset Class
# =======================
class FallSequenceDataset(Dataset):
    def __init__(self, sequence_label_list, transform=None):
        self.data = sequence_label_list
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        frame_paths, label = self.data[idx]
        frames = []
        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            frames.append(img)
        # Output shape [C,T,H,W]
        frames_tensor = torch.stack(frames).permute(1,0,2,3)
        return frames_tensor, torch.tensor(label, dtype=torch.long)

train_dataset = FallSequenceDataset(train_samples)
val_dataset = FallSequenceDataset(val_samples)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

# =======================
# Model: 3D CNN + ConvLSTM2D
# =======================
from convlstm import ConvLSTM

class FallDetector3D(nn.Module):
    def __init__(self, num_classes=3):
        super(FallDetector3D, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(96, 192, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),

            nn.Conv3d(192, 384, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(384),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2))
        )
        self.convlstm = ConvLSTM(input_dim=384,
                                 hidden_dim=384,
                                 kernel_size=(3,3),
                                 num_layers=2,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)
        self.pool = nn.AdaptiveAvgPool2d((7,7))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(384*7*7, num_classes)

    def forward(self, x):
        feat3d = self.conv3d(x)               # [B,384,T,H,W]
        B, C, T, H, W = feat3d.shape
        feat3d = feat3d.permute(0,2,1,3,4)    # [B,T,C,H,W]
        out, _ = self.convlstm(feat3d)
        last_out = out[0][:,-1]               # [B,C,H,W]
        pooled = self.pool(last_out)          # [B,C,7,7]
        flat = pooled.view(B, -1)
        flat = self.dropout(flat)
        return self.fc(flat)

# =======================
# Training with Early Stopping + History
# =======================
start_time = time.time()
os.makedirs("./3dcnn_convlstm2d_Wmodels/2cams", exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FallDetector3D().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)#0.00005

EPOCHS = 50
patience = 10
patience_counter = 0
best_val_loss = float('inf')

best_model_path = "./3dcnn_convlstm2d_Wmodels/2cams/fall_3dcnn_convlstm2d_best-patience5.pth"
final_model_path = "./3dcnn_convlstm2d_Wmodels/2cams/fall_3dcnn_convlstm2d_final-patience5.pth"

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(EPOCHS):
    # Training loop
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = torch.max(out, 1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    train_acc = correct / total
    train_loss = running_loss / len(train_loader)

    # Validation loop
    model.eval()
    v_correct, v_total, v_loss = 0, 0, 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            v_loss += loss.item()
            _, pred = torch.max(out, 1)
            v_correct += (pred == y).sum().item()
            v_total += y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    val_acc = v_correct / v_total
    val_loss = v_loss / len(val_loader)

    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss {train_loss:.4f} Acc {train_acc:.4f}")
    print(f"Val   Loss {val_loss:.4f} Acc {val_acc:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at epoch {epoch+1}")
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

    # Optional: confusion matrix and report
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ADL","Fall","Lie Down"], yticklabels=["ADL","Fall","Lie Down"])
    plt.show()

    print(classification_report(all_labels, all_preds,
                                labels=[0,1,2],
                                target_names=["ADL","Fall","Lie Down"]))

# Save final model and history
torch.save(model.state_dict(), final_model_path)
with open("./3dcnn_convlstm2d_Wmodels/2cams/training_history-patience5.pkl", "wb") as f:
    pickle.dump(history, f)

print(f"Final model saved to {final_model_path}")
print("Training history saved for plotting.")
end_time = time.time()
elapsed_time = end_time - start_time
print(f" Time: {elapsed_time:.6f} sec")
