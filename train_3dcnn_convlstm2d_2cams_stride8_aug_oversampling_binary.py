import os, json, random, time, pickle
import numpy as np
from collections import Counter, defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =============== Reproducibility ===============
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============== Hyperparameters (single source of truth) ===============
HP = {
    "batch_size": 8,
    "epochs": 50,
    "patience": 20,
    "lr": 5e-5,
    "seq_len": 16,
    "cuda_visible": "0",  # GPU index as string
    "project": "fall-detection",
    "run_name": "3dcnn-convlstm-2cams-stride8-aug-oversample-BINARY",

    # augmentation / normalization switches
    "use_normalize": True,                             # apply identical Normalize to train & val
    "normalize_mean": [0.457405, 0.47059, 0.444467],   # dataset-specific mean (RGB in [0,1])
    "normalize_std":  [0.216927, 0.210016, 0.198048],  # dataset-specific std  (RGB in [0,1])
    "fall_aug": False,                                 # enable class-conditional aug for Fall (label=1)
    "fall_aug_p": 0.6,                                 # probability to APPLY aug on Fall frames
}

# =============== Load dataset (BINARY labels: 0=Non-Fall, 1=Fall) ===============
# Prefer the new binary pickle; if not found, fallback to the old name.
dataset_path_primary = "./f_seq_stride8_binary.pkl"
dataset_path_fallback = "./f_seq_stride8_binary.pkl"

if os.path.exists(dataset_path_primary):
    ds_path = dataset_path_primary
elif os.path.exists(dataset_path_fallback):
    ds_path = dataset_path_fallback
else:
    raise FileNotFoundError("Neither f_seq_stride8_binary.pkl nor f_seq_stride8.pkl was found.")

with open(ds_path, "rb") as f:
    dataset = pickle.load(f)

print(f"Loaded dataset from {ds_path} with {len(dataset)} sequences.")

# Expect only labels {0,1}; if any {2} existed in source, they should already be mapped to 0 in your binary export.
label_counts = Counter(y for _, y in dataset)
print(f"Class counts (0=Non-Fall, 1=Fall): {dict(label_counts)}")

# =============== Actor-based split (deterministic) ===============
# We split by actor identity to avoid identity leakage between train/val.
actor_to_samples = {}
for seq_paths, label in dataset:
    parts = seq_paths[0].split(os.sep)
    # Assumes a folder name like "Actor_XX" exists somewhere in the path
    actor_folder = next(p for p in parts if p.startswith("Actor_"))
    actor_id = "_".join(actor_folder.split("_")[:2])  # "Actor_XX"
    actor_to_samples.setdefault(actor_id, []).append((seq_paths, label))

all_actors = sorted(actor_to_samples.keys())
rng = random.Random(SEED); rng.shuffle(all_actors)
split_idx = int(0.8 * len(all_actors))
train_actors, val_actors = all_actors[:split_idx], all_actors[split_idx:]
train_samples = [s for a in train_actors for s in actor_to_samples[a]]
val_samples   = [s for a in val_actors   for s in actor_to_samples[a]]

print("VAL actors:", val_actors)
print("TRAIN actors:", train_actors)
print("VAL class dist (0=Non-Fall,1=Fall):", Counter([y for _, y in val_samples]))
print("TRAIN class dist (before oversample):", Counter([y for _, y in train_samples]))

# =============== Oversample minority to the majority count ===============
# In binary setup, we upsample the minority (usually Fall) to match the majority.
by_class = defaultdict(list)
for s in train_samples:
    by_class[s[1]].append(s)

target_per_class = max(len(v) for v in by_class.values())
print("Target per class (majority size):", target_per_class)

balanced = []
for c, samples_c in by_class.items():
    n = len(samples_c)
    if n >= target_per_class:
        balanced.extend(samples_c)
    else:
        # replicate entire list + random remainder
        times = target_per_class // n
        rem   = target_per_class - times * n
        balanced.extend(samples_c * times + rng.sample(samples_c, rem))

train_samples = balanced
rng.shuffle(train_samples)

print("TRAIN class dist (after oversample):", Counter([y for _, y in train_samples]))
print(f"Train seqs:{len(train_samples)}  Val seqs:{len(val_samples)}")

# =============== W&B Init ===============
import wandb
run = wandb.init(
    project=HP["project"],
    name=HP["run_name"],
    tags=["actor-split","2cams","stride8","oversample","mix-aug","BINARY"],
    config=HP,
)
cfg = wandb.config

# =============== Transforms (train vs. val) ===============
# IMPORTANT: If normalization is enabled, keep it IDENTICAL for train & val.
norm_tf = []
if cfg.use_normalize:
    norm_tf = [transforms.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std)]

# Base transform applied to ALL classes (train & val)
base_tf_train = transforms.Compose([
    transforms.ToTensor(),
    *norm_tf,
])

base_tf_val = transforms.Compose([
    transforms.ToTensor(),
    *norm_tf,
])

# Stronger transform used conditionally for Fall (label=1) in training
fall_tf_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
    transforms.ToTensor(),
    *norm_tf,  # keep normalization identical
])

# =============== Dataset / Loader ===============
class FallSequenceDataset(Dataset):
    """
    Returns a clip as [C,T,H,W] and a binary label (0=Non-Fall, 1=Fall).
    - Temporal jitter in training: choose a random contiguous subsequence of length seq_len.
    - For Fall (label=1), optionally apply augmentation with probability p (cfg.fall_aug_p).
      This MIXES augmented and original Fall samples, improving robustness vs. 100% augmentation.
    """
    def __init__(self, sequence_label_list, seq_len=16, train=False):
        self.data = sequence_label_list
        self.seq_len = int(seq_len)
        self.train = train
        self.p_aug = float(getattr(cfg, "fall_aug_p", 0.6))  # default 0.6 if missing

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        frame_paths, label = self.data[idx]

        # Temporal jitter only for training (if sequence is long enough)
        if self.train and len(frame_paths) > self.seq_len:
            start = random.randint(0, len(frame_paths) - self.seq_len)
            paths = frame_paths[start:start + self.seq_len]
        else:
            paths = frame_paths[:self.seq_len]

        frames = []
        for p in paths:
            img = Image.open(p).convert("RGB")

            if self.train:
                if label == 1 and cfg.fall_aug:
                    # With probability p -> augmented view; else -> original (normalized)
                    if random.random() < self.p_aug:
                        img = fall_tf_train(img)
                    else:
                        img = base_tf_train(img)
                else:
                    img = base_tf_train(img)
            else:
                img = base_tf_val(img)

            frames.append(img)

        # Stack to [T,C,H,W] then permute to [C,T,H,W]
        x = torch.stack(frames).permute(1,0,2,3)
        return x, torch.tensor(label, dtype=torch.long)

train_ds = FallSequenceDataset(train_samples, seq_len=cfg.seq_len, train=True)
val_ds   = FallSequenceDataset(val_samples,   seq_len=cfg.seq_len, train=False)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

# =============== Model ===============
from convlstm import ConvLSTM

class FallDetector3D(nn.Module):
    """3D CNN feature extractor + ConvLSTM head + GAP + FC for BINARY classification."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3,96,3,1,1), nn.BatchNorm3d(96), nn.ReLU(), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(96,192,3,1,1), nn.BatchNorm3d(192), nn.ReLU(), nn.MaxPool3d((2,2,2)),
            nn.Conv3d(192,384,3,1,1), nn.BatchNorm3d(384), nn.ReLU(), nn.MaxPool3d((2,2,2)),
        )
        self.convlstm = ConvLSTM(
            input_dim=384, hidden_dim=384, kernel_size=(3,3),
            num_layers=2, batch_first=True, bias=True, return_all_layers=False
        )
        self.pool = nn.AdaptiveAvgPool2d((7,7))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(384*7*7, num_classes)

    def forward(self, x):
        # x: [B, C, T, H, W]
        f = self.conv3d(x)             # [B,384,T,H,W]
        B,C,T,H,W = f.shape
        f = f.permute(0,2,1,3,4)       # [B,T,C,H,W]
        out,_ = self.convlstm(f)
        last = out[0][:,-1]            # [B,C,H,W]
        pooled = self.pool(last)       # [B,C,7,7]
        flat = pooled.view(B,-1)
        flat = self.dropout(flat)
        return self.fc(flat)           # logits: [B,2]

# =============== Device / Optimizer / Loss ===============
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FallDetector3D(num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

# With oversampling, plain CrossEntropyLoss is appropriate (2 logits -> softmax over {0,1}).
criterion = nn.CrossEntropyLoss()

# Optional: log params & grads to W&B
wandb.watch(model, criterion, log="all", log_freq=100)

# =============== Training Loop ===============
EPOCHS = cfg.epochs
PATIENCE = cfg.patience
best_val_loss = float("inf")
patience_counter = 0

best_model_path  = "./3dcnn_convlstm2d_Wmodels/2cams/best_binary.pth"
final_model_path = "./3dcnn_convlstm2d_Wmodels/2cams/final_binary.pth"
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
start_time = time.time()

for epoch in range(EPOCHS):
    # -------- Train --------
    model.train()
    total=0; correct=0; run_loss=0.0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)                 # logits [B,2]
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        correct += (out.argmax(1)==y).sum().item()
        total += y.size(0)
    train_loss = run_loss/len(train_loader)
    train_acc  = correct/total

    # -------- Validate --------
    model.eval()
    v_total=0; v_correct=0; v_loss_sum=0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out,y)
            v_loss_sum += loss.item()
            pred = out.argmax(1)
            v_correct += (pred==y).sum().item()
            v_total += y.size(0)
            all_preds.extend(pred.cpu().numpy()); all_labels.extend(y.cpu().numpy())
    val_loss = v_loss_sum/len(val_loader)
    val_acc  = v_correct/v_total

    # -------- Log history + W&B --------
    history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc);   history['val_acc'].append(val_acc)
    wandb.log({"epoch": epoch+1, "train/loss":train_loss, "train/acc":train_acc,
               "val/loss":val_loss, "val/acc":val_acc, "lr": optimizer.param_groups[0]['lr']})

    print(f"\nEpoch {epoch+1}/{EPOCHS} | Train {train_loss:.4f}/{train_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f}")

    # -------- Save best checkpoint --------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        art = wandb.Artifact("best-model-binary", type="model")
        art.add_file(best_model_path); wandb.log_artifact(art)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # -------- Confusion matrix (per epoch, binary) --------
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1])
    fig = plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Fall","Fall"], yticklabels=["Non-Fall","Fall"])
    plt.title(f"Confusion Matrix (epoch {epoch+1})"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    wandb.log({"val/confusion_matrix": wandb.Image(fig)}); plt.close(fig)

    # -------- Classification report (per epoch, binary) --------
    report = classification_report(
        all_labels, all_preds,
        labels=[0, 1],
        target_names=["Non-Fall", "Fall"],
        output_dict=True
    )
    print(classification_report(
        all_labels, all_preds,
        labels=[0, 1],
        target_names=["Non-Fall", "Fall"]
    ))
    wandb.log({
        "val/NonFall/precision": report["Non-Fall"]["precision"],
        "val/NonFall/recall":    report["Non-Fall"]["recall"],
        "val/NonFall/f1":        report["Non-Fall"]["f1-score"],
        "val/Fall/precision":    report["Fall"]["precision"],
        "val/Fall/recall":       report["Fall"]["recall"],
        "val/Fall/f1":           report["Fall"]["f1-score"],
        "val/accuracy":          report["accuracy"],
    })

# =============== Save final model & history ===============
torch.save(model.state_dict(), final_model_path)
with open("./3dcnn_convlstm2d_Wmodels/2cams/training_history_binary.pkl", "wb") as f:
    pickle.dump(history, f)

elapsed = time.time()-start_time
print(f"Final saved to {final_model_path} | Time: {elapsed:.2f}s")
wandb.finish()
