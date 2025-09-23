import os, json, random, time, pickle
import numpy as np
from collections import Counter
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

# =============== Single source of truth for HPs ===============
HP = {
    "batch_size": 8,
    "epochs": 50,
    "patience": 20,
    "lr": 5e-5,
    "seq_len": 16,
    "cuda_visible": "0",  # GPU index as string
    "project": "fall-detection",
    "run_name": "3dcnn-convlstm-2cams-stride8-aug-cw-norm",

    # augmentation / normalization switches
    "use_normalize": True,                      # if True -> apply same Normalize in train & val
    "normalize_mean": [0.485, 0.456, 0.406],    # ImageNet-like stats (adjust if you have your own)
    "normalize_std":  [0.229, 0.224, 0.225],
    "fall_aug": True,                           # enable stronger aug for Fall class (label=1)
}

# =============== Data prep ===============
with open("./f_seq_stride8.pkl", "rb") as f:
    dataset = pickle.load(f)
print(f"Loaded dataset with {len(dataset)} sequences.")
adl = sum(1 for _, y in dataset if y==0); fall = sum(1 for _, y in dataset if y==1); lie = sum(1 for _, y in dataset if y==2)
print(f"ADL:{adl}  Fall:{fall}  Lie:{lie}")

# Actor split (deterministic)
actor_to_samples = {}
for seq_paths, label in dataset:
    parts = seq_paths[0].split(os.sep)
    actor_folder = next(p for p in parts if p.startswith("Actor_"))
    actor_id = "_".join(actor_folder.split("_")[:2])
    actor_to_samples.setdefault(actor_id, []).append((seq_paths, label))

all_actors = sorted(actor_to_samples.keys())
rng = random.Random(SEED); rng.shuffle(all_actors)
split_idx = int(0.8 * len(all_actors))
train_actors, val_actors = all_actors[:split_idx], all_actors[split_idx:]
train_samples = [s for a in train_actors for s in actor_to_samples[a]]
val_samples   = [s for a in val_actors   for s in actor_to_samples[a]]
print("VAL actors:", val_actors)
print("TRAIN actors:", train_actors)
print("VAL class dist:", Counter([y for _, y in val_samples]))
print("TRAIN class dist:", Counter([y for _, y in train_samples]))
print(f"Train seqs:{len(train_samples)}  Val seqs:{len(val_samples)}")

# =============== W&B ===============
import wandb
run = wandb.init(
    project=HP["project"],
    name=HP["run_name"],
    tags=["actor-split","2cams","stride8"],
    config=HP,
)
cfg = wandb.config  # always read hyperparams from wandb.config

# =============== Transforms (train vs. val) ===============
# NOTE: If normalization is enabled, it must be identical in train & val.
norm_tf = []
if cfg.use_normalize:
    norm_tf = [transforms.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std)]

# Base transform applied to ALL classes (train & val)
base_tf_train = transforms.Compose([
    # transforms.Resize((224,224)),         # enable if your model expects resized inputs
    transforms.ToTensor(),
    *norm_tf,
])

base_tf_val = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.ToTensor(),
    *norm_tf,
])

# Stronger transform ONLY for Fall (label=1) during training
fall_tf_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
    transforms.ToTensor(),
    *norm_tf,  # same normalization as base
])

# =============== Dataset / Loader ===============
class FallSequenceDataset(Dataset):
    """Returns a clip as [C,T,H,W] and the label. Uses temporal jitter in training."""
    def __init__(self, sequence_label_list, seq_len=16, train=False):
        self.data = sequence_label_list
        self.seq_len = int(seq_len)
        self.train = train

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        frame_paths, label = self.data[idx]

        # --- temporal jitter for training (choose random contiguous subsequence) ---
        if self.train and len(frame_paths) > self.seq_len:
            start = random.randint(0, len(frame_paths) - self.seq_len)
            paths = frame_paths[start:start + self.seq_len]
        else:
            paths = frame_paths[:self.seq_len]

        frames = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            if self.train and cfg.fall_aug and label == 1:
                # class-conditional augmentation (stronger for Fall)
                img = fall_tf_train(img)
            else:
                # base transform (same normalization as val)
                img = base_tf_train(img) if self.train else base_tf_val(img)
            frames.append(img)

        x = torch.stack(frames).permute(1,0,2,3)  # [C,T,H,W]
        return x, torch.tensor(label, dtype=torch.long)

train_ds = FallSequenceDataset(train_samples, seq_len=cfg.seq_len, train=True)
val_ds   = FallSequenceDataset(val_samples,   seq_len=cfg.seq_len, train=False)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

# =============== Model ===============
from convlstm import ConvLSTM
class FallDetector3D(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3,96,3,1,1), nn.BatchNorm3d(96), nn.ReLU(), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(96,192,3,1,1), nn.BatchNorm3d(192), nn.ReLU(), nn.MaxPool3d((2,2,2)),
            nn.Conv3d(192,384,3,1,1), nn.BatchNorm3d(384), nn.ReLU(), nn.MaxPool3d((2,2,2)),
        )
        self.convlstm = ConvLSTM(384, 384, (3,3), num_layers=2, batch_first=True, bias=True, return_all_layers=False)
        self.pool = nn.AdaptiveAvgPool2d((7,7))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(384*7*7, 3)
    def forward(self, x):
        f = self.conv3d(x)            # [B,384,T,H,W]
        B,C,T,H,W = f.shape
        f = f.permute(0,2,1,3,4)      # [B,T,C,H,W]
        out,_ = self.convlstm(f)
        last = out[0][:,-1]           # [B,C,H,W]
        pooled = self.pool(last)
        flat = pooled.view(B,-1)
        flat = self.dropout(flat)
        return self.fc(flat)

# =============== Device / Optim / Class Weights ===============
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FallDetector3D().to(device)
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

# ---- compute class weights from TRAIN labels (inverse-frequency) ----
train_labels = [y for _, y in train_samples]
cls_counts = np.bincount(train_labels, minlength=3)
eps = 1e-8
cls_weights = cls_counts.sum() / (len(cls_counts) * (cls_counts + eps))  # mean_count / class_count
print("Class counts (train):", cls_counts, " -> weights:", cls_weights)

weights_tensor = torch.tensor(cls_weights, dtype=torch.float, device=device)
# Use class-weighted CE (you can add mild label smoothing if desired)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
# criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.05)

# Optional: watch grads/params
wandb.watch(model, criterion, log="all", log_freq=100)

# =============== Training ===============
EPOCHS = cfg.epochs
PATIENCE = cfg.patience
best_val_loss = float("inf")
patience_counter = 0

best_model_path  = "./3dcnn_convlstm2d_Wmodels/2cams/best.pth"
final_model_path = "./3dcnn_convlstm2d_Wmodels/2cams/final.pth"
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
start_time = time.time()

for epoch in range(EPOCHS):
    # --- train ---
    model.train()
    total=0; correct=0; run_loss=0.0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        correct += (out.argmax(1)==y).sum().item()
        total += y.size(0)
    train_loss = run_loss/len(train_loader)
    train_acc  = correct/total

    # --- val ---
    model.eval()
    v_total=0; v_correct=0; v_loss_sum=0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x,y in val_loader:
            x,y=x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out,y)
            v_loss_sum += loss.item()
            pred = out.argmax(1)
            v_correct += (pred==y).sum().item()
            v_total += y.size(0)
            all_preds.extend(pred.cpu().numpy()); all_labels.extend(y.cpu().numpy())
    val_loss = v_loss_sum/len(val_loader)
    val_acc  = v_correct/v_total

    # history + wandb
    history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc);   history['val_acc'].append(val_acc)
    wandb.log({"epoch": epoch+1, "train/loss":train_loss, "train/acc":train_acc,
               "val/loss":val_loss, "val/acc":val_acc, "lr": optimizer.param_groups[0]['lr']})

    print(f"\nEpoch {epoch+1}/{EPOCHS} | Train {train_loss:.4f}/{train_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f}")

    # save best + artifact
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        art = wandb.Artifact("best-model", type="model"); art.add_file(best_model_path); wandb.log_artifact(art)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # (optional) log confusion matrix image per epoch
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
    fig = plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ADL","Fall","Lie Down"], yticklabels=["ADL","Fall","Lie Down"])
    plt.title(f"Confusion Matrix (epoch {epoch+1})"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    wandb.log({"val/confusion_matrix": wandb.Image(fig)}); plt.close(fig)

    # --- classification report (per epoch) ---
    report = classification_report(
        all_labels,
        all_preds,
        labels=[0, 1, 2],
        target_names=["ADL", "Fall", "Lie Down"],
        output_dict=True
    )
    
    # print nicely in terminal (text)
    print(classification_report(
        all_labels,
        all_preds,
        labels=[0, 1, 2],
        target_names=["ADL", "Fall", "Lie Down"]
    ))
    
    # log class-wise metrics to W&B with hierarchical keys
    wandb.log({
        # ADL
        "val/ADL/precision":   report["ADL"]["precision"],
        "val/ADL/recall":      report["ADL"]["recall"],
        "val/ADL/f1":          report["ADL"]["f1-score"],
    
        # Fall
        "val/Fall/precision":  report["Fall"]["precision"],
        "val/Fall/recall":     report["Fall"]["recall"],
        "val/Fall/f1":         report["Fall"]["f1-score"],
    
        # Lie Down
        "val/LieDown/precision": report["Lie Down"]["precision"],
        "val/LieDown/recall":    report["Lie Down"]["recall"],
        "val/LieDown/f1":        report["Lie Down"]["f1-score"],
    
        # overall accuracy
        "val/accuracy": report["accuracy"],
    })


# =============== Save final & finish ===============
torch.save(model.state_dict(), final_model_path)
with open("./3dcnn_convlstm2d_Wmodels/2cams/training_history.pkl", "wb") as f:
    pickle.dump(history, f)
elapsed = time.time()-start_time
print(f"Final saved to {final_model_path} | Time: {elapsed:.2f}s")
wandb.finish()
