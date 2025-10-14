#pip install timm huggingface_hub
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

# ----------------- New imports for UniFormer -----------------
from huggingface_hub import hf_hub_download
from uniformer import uniformer_small600   # make sure uniformer.py is available

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
    "seq_len": 16,               # UniFormer Kinetics weights expect 16 frames (W16)
    "cuda_visible": "0",         # GPU index as string
    "project": "fall-detection",
    "run_name": "uniformer_small_k600_16x8-2cams-stride8-oversample",

    # augmentation / normalization switches
    # NOTE: For best transfer with ImageNet-pretrained weights, use ImageNet stats
    "use_normalize": True,
    "normalize_mean": [0.485, 0.456, 0.406],  # ImageNet mean
    "normalize_std":  [0.229, 0.224, 0.225],  # ImageNet std
    "fall_aug": False,                         # class-conditional aug for Fall (label=1)
    "fall_aug_p": 0.6,                        # prob to APPLY aug on Fall frames (mix aug + original)
}

# =============== Load dataset ===============
with open("./f_seq_stride8.pkl", "rb") as f:
    dataset = pickle.load(f)
print(f"Loaded dataset with {len(dataset)} sequences.")
adl = sum(1 for _, y in dataset if y==0); fall = sum(1 for _, y in dataset if y==1); lie = sum(1 for _, y in dataset if y==2)
print(f"ADL:{adl}  Fall:{fall}  Lie:{lie}")

# =============== Actor-based split (deterministic) ===============
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
print("TRAIN class dist (before oversample):", Counter([y for _, y in train_samples]))

# =============== Oversample minorities to majority size ===============
# Goal: upsample minority classes (e.g., Fall) to match the largest class count
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
        # replicate full lists + random remainder to reach target_per_class
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
    tags=["actor-split","2cams","stride8","oversample","uniformer"],
    config=HP,
)
cfg = wandb.config

# =============== Transforms (train vs. val) ===============
# IMPORTANT: UniFormer Kinetics weights were trained on 224x224, ImageNet normalization.
norm_tf = []
if cfg.use_normalize:
    norm_tf = [transforms.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std)]

# Base transform applied to ALL classes (train & val)
base_tf_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    *norm_tf,
])

base_tf_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    *norm_tf,
])

# Stronger transform used conditionally for Fall (label=1) in training
fall_tf_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
    transforms.ToTensor(),
    *norm_tf,  # keep normalization identical
])

# =============== Dataset / Loader ===============
class FallSequenceDataset(Dataset):
    """
    Returns a clip as [C,T,H,W] and the label.
    - Temporal jitter in training: choose a random contiguous subsequence of length seq_len.
    - For Fall (label=1), apply augmentation with probability p (cfg.fall_aug_p). Otherwise keep original (normalized).
      => Mixing augmented and original Fall samples is usually more stable than 100% augmented.
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

        # Stack to [T,C,H,W] then permute to [C,T,H,W] (UniFormer expects [B,C,T,H,W])
        x = torch.stack(frames).permute(1,0,2,3)
        return x, torch.tensor(label, dtype=torch.long)

train_ds = FallSequenceDataset(train_samples, seq_len=cfg.seq_len, train=True)
val_ds   = FallSequenceDataset(val_samples,   seq_len=cfg.seq_len, train=False)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

# =============== UniFormer Model ===============
class TransformerFallDetector(nn.Module):
    """
    UniFormer (video transformer) backbone + new classification head for 3 classes.
    Pretrained on Kinetics-400 with window 16x8 (16 frames, temporal stride 8).
    Input shape: [B, 3, 16, 224, 224]
    """
    def __init__(self, num_classes=3, pretrained_variant="small_k600_16x8"):
        super().__init__()
        # 1) Build backbone (Uniformer small)
        self.backbone = uniformer_small600()

        # 2) Load pretrained weights that match the chosen backbone
        #    Available filenames in Sense-X/uniformer_video include:
        #    uniformer_base_k400_16x8.pth, uniformer_base_k400_8x8.pth, uniformer_base_k600_16x8.pth, etc.
        repo = "Sense-X/uniformer_video"
        fname = "uniformer_small_k600_16x8.pth"
        ckpt_path = hf_hub_download(repo_id=repo, filename=fname)
        state = torch.load(ckpt_path, map_location="cpu")
        self.backbone.load_state_dict(state)

        # 3) Replace classifier (head) with the number of classes in our dataset
        #    The base model's last embedding dim is 512.
        self.backbone.embed_dim = 512
        self.backbone.reset_classifier(num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

# =============== Device / Optimizer / Loss ===============
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerFallDetector(num_classes=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

# After oversampling, plain CrossEntropyLoss is fine (no class weights)
criterion = nn.CrossEntropyLoss()

# Optional: log params & grads to W&B
wandb.watch(model, criterion, log="all", log_freq=100)

# =============== Training Loop ===============
EPOCHS = cfg.epochs
PATIENCE = cfg.patience
best_val_loss = float("inf")
patience_counter = 0

# Save paths (renamed for transformer runs)
best_model_path  = "./uniformer_Wmodels/2cams/best.pth"
final_model_path = "./uniformer_Wmodels/2cams/final.pth"
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
start_time = time.time()

for epoch in range(EPOCHS):
    # -------- Train --------
    model.train()
    total=0; correct=0; run_loss=0.0
    for x,y in train_loader:
        # x: [B,C,T,H,W]
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
        art = wandb.Artifact("best-model", type="model")
        art.add_file(best_model_path); wandb.log_artifact(art)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # -------- Confusion matrix (per epoch) --------
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
    fig = plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ADL","Fall","Lie Down"], yticklabels=["ADL","Fall","Lie Down"])
    plt.title(f"Confusion Matrix (epoch {epoch+1})"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    wandb.log({"val/confusion_matrix": wandb.Image(fig)}); plt.close(fig)

    # -------- Classification report (per epoch) --------
    report = classification_report(
        all_labels, all_preds,
        labels=[0, 1, 2],
        target_names=["ADL", "Fall", "Lie Down"],
        output_dict=True
    )
    print(classification_report(
        all_labels, all_preds,
        labels=[0, 1, 2],
        target_names=["ADL", "Fall", "Lie Down"]
    ))
    wandb.log({
        "val/ADL/precision":     report["ADL"]["precision"],
        "val/ADL/recall":        report["ADL"]["recall"],
        "val/ADL/f1":            report["ADL"]["f1-score"],
        "val/Fall/precision":    report["Fall"]["precision"],
        "val/Fall/recall":       report["Fall"]["recall"],
        "val/Fall/f1":           report["Fall"]["f1-score"],
        "val/LieDown/precision": report["Lie Down"]["precision"],
        "val/LieDown/recall":    report["Lie Down"]["recall"],
        "val/LieDown/f1":        report["Lie Down"]["f1-score"],
        "val/accuracy":          report["accuracy"],
    })

# =============== Save final model & history ===============
torch.save(model.state_dict(), final_model_path)
with open("./uniformer_Wmodels/2cams/training_history.pkl", "wb") as f:
    pickle.dump(history, f)

elapsed = time.time()-start_time
print(f"Final saved to {final_model_path} | Time: {elapsed:.2f}s")
wandb.finish()
