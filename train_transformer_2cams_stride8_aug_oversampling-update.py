# pip install timm huggingface_hub wandb

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

# ----------------- UniFormer imports -----------------
from huggingface_hub import hf_hub_download
from uniformer import uniformer_small600   # backbone with 600-class head (matches K600 ckpt)

# =============== Reproducibility ===============
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============== Hyperparameters ===============
HP = {
    "batch_size": 8,
    "epochs": 50,
    "patience": 10,
    "lr": 1e-5,
    "seq_len": 16,               # UniFormer pretrained weights work great with 16 frames
    "cuda_visible": "0",
    "project": "fall-detection",
    "run_name": "uniformer_small_k600_16x8-2cams-stride8-oversample-T&VconfM-cutrepo-correctLogic",

    # transforms / normalization
    "use_normalize": True,
    "normalize_mean": [0.485, 0.456, 0.406],  # ImageNet mean
    "normalize_std":  [0.229, 0.224, 0.225],  # ImageNet std
    "fall_aug": False,                         # optional class-conditional aug for Fall (label=1)
    "fall_aug_p": 0.6,                         # prob to apply aug when fall_aug=True
}

# =============== Load dataset ===============
with open("./seq_official_len16_stride8_new.pkl", "rb") as f:
    dataset = pickle.load(f)
print(f"Loaded dataset with {len(dataset)} sequences.")
adl = sum(1 for _, y in dataset if y == 0)
fall = sum(1 for _, y in dataset if y == 1)
lie = sum(1 for _, y in dataset if y == 2)
print(f"ADL:{adl}  Fall:{fall}  Lie:{lie}")

# =============== Actor-based split ===============
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

# =============== Oversampling (train only) ===============
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
    tags=["actor-split","2cams","stride8","oversample","uniformer","cut_repositioning"],
    config=HP,
)

# Make epoch the global x-axis
wandb.define_metric("epoch")                    # epoch is a recorded metric
wandb.define_metric("train/*", step_metric="epoch")
wandb.define_metric("val/*",   step_metric="epoch")
wandb.define_metric("lr",      step_metric="epoch")

cfg = wandb.config

# =============== Transforms ===============
norm_tf = []
if cfg.use_normalize:
    norm_tf = [transforms.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std)]

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

fall_tf_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
    transforms.ToTensor(),
    *norm_tf,
])

# =============== Dataset / Loader ===============
class FallSequenceDataset(Dataset):
    """
    Returns a clip as [C,T,H,W] and the label.
    - Temporal jitter in training: pick random contiguous subsequence of length seq_len.
    - Optional class-conditional aug for Fall with probability p.
    """
    def __init__(self, sequence_label_list, seq_len=16, train=False):
        self.data = sequence_label_list
        self.seq_len = int(seq_len)
        self.train = train
        self.p_aug = float(getattr(cfg, "fall_aug_p", 0.6))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_paths, label = self.data[idx]

        # temporal crop / jitter
        if self.train and len(frame_paths) > self.seq_len:
            start = random.randint(0, len(frame_paths) - self.seq_len)
            paths = frame_paths[start:start + self.seq_len]
        else:
            paths = frame_paths[:self.seq_len]

        frames = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            if self.train:
                if label == 1 and cfg.fall_aug and random.random() < self.p_aug:
                    img = fall_tf_train(img)
                else:
                    img = base_tf_train(img)
            else:
                img = base_tf_val(img)
            frames.append(img)

        # stack to [T,C,H,W] then permute to [C,T,H,W]
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
    Pretrained weights: uniformer_small_k600_16x8.pth
    Input shape: [B, 3, 16, 224, 224]
    """
    def __init__(self, num_classes=3):
        super().__init__()
        # Build backbone with a 600-class head so checkpoint loads cleanly
        self.backbone = uniformer_small600()

        # Load pretrained weights (Kinetics-600, 16x8)
        repo = "Sense-X/uniformer_video"
        fname = "uniformer_small_k600_16x8.pth"
        ckpt_path = hf_hub_download(repo_id=repo, filename=fname)
        state = torch.load(ckpt_path, map_location="cpu")
        self.backbone.load_state_dict(state)

        # Replace classifier head for our task
        self.backbone.embed_dim = 512
        self.backbone.reset_classifier(num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

# =============== Device / Optimizer / Loss ===============
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerFallDetector(num_classes=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
criterion = nn.CrossEntropyLoss()

# Optional: log params & grads to W&B
wandb.watch(model, criterion, log="all", log_freq=100)

# =============== Training Loop ===============
EPOCHS = cfg.epochs
PATIENCE = cfg.patience
best_val_loss = float("inf")
patience_counter = 0

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

    # -------- Train confusion matrix (computed after training step) --------
    model.eval()
    train_all_preds, train_all_labels = [], []
    with torch.no_grad():
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            train_all_preds.extend(preds.cpu().numpy())
            train_all_labels.extend(y.cpu().numpy())
    train_cm = confusion_matrix(train_all_labels, train_all_preds, labels=[0,1,2])
    fig_train = plt.figure(figsize=(6,5))
    sns.heatmap(train_cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["ADL","Fall","Lie Down"], yticklabels=["ADL","Fall","Lie Down"])
    plt.title(f"Train Confusion Matrix (epoch {epoch+1})")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    wandb.log({"epoch": epoch+1, "train/confusion_matrix": wandb.Image(fig_train)}, step=epoch+1)
    plt.close(fig_train)

    # -------- Validate --------
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

    # -------- Log scalars (force epoch as x-axis) --------
    history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc);   history['val_acc'].append(val_acc)
    wandb.log({
        "epoch": epoch+1,
        "train/loss": train_loss,
        "train/acc":  train_acc,
        "val/loss":   val_loss,
        "val/acc":    val_acc,
        "lr": optimizer.param_groups[0]['lr'],
    }, step=epoch+1)

    print(f"\nEpoch {epoch+1}/{EPOCHS} | Train {train_loss:.4f}/{train_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f}")

    # -------- Save best checkpoint & (NEW) print TRAIN classification report at best --------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        # Save checkpoint
        torch.save(model.state_dict(), best_model_path)
        art = wandb.Artifact("best-model", type="model")
        art.add_file(best_model_path); wandb.log_artifact(art)

        # (NEW) Print & log TRAIN classification report ONLY at best model
        train_report = classification_report(
            train_all_labels, train_all_preds,
            labels=[0,1,2],
            target_names=["ADL","Fall","Lie Down"],
            output_dict=True
        )
        # Pretty print to terminal
        print("\n=== TRAIN classification report @ BEST MODEL ===")
        print(classification_report(
            train_all_labels, train_all_preds,
            labels=[0,1,2],
            target_names=["ADL","Fall","Lie Down"]
        ))
        # Log main train metrics at best (scalars)
        wandb.log({
            "epoch": epoch+1,
            "train_best/ADL/precision":     train_report["ADL"]["precision"],
            "train_best/ADL/recall":        train_report["ADL"]["recall"],
            "train_best/ADL/f1":            train_report["ADL"]["f1-score"],
            "train_best/Fall/precision":    train_report["Fall"]["precision"],
            "train_best/Fall/recall":       train_report["Fall"]["recall"],
            "train_best/Fall/f1":           train_report["Fall"]["f1-score"],
            "train_best/LieDown/precision": train_report["Lie Down"]["precision"],
            "train_best/LieDown/recall":    train_report["Lie Down"]["recall"],
            "train_best/LieDown/f1":        train_report["Lie Down"]["f1-score"],
            "train_best/accuracy":          train_report["accuracy"],
        }, step=epoch+1)

    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            # optional: break here
            break

    # -------- Val confusion matrix --------
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
    fig = plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ADL","Fall","Lie Down"], yticklabels=["ADL","Fall","Lie Down"])
    plt.title(f"Validation Confusion Matrix (epoch {epoch+1})"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    wandb.log({"epoch": epoch+1, "val/confusion_matrix": wandb.Image(fig)}, step=epoch+1)
    plt.close(fig)

    # -------- Val classification report (per epoch) --------
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
        "epoch": epoch+1,
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
    }, step=epoch+1)

# =============== Save final model & history ===============
final_model_path = "./uniformer_Wmodels/2cams/final.pth"
torch.save(model.state_dict(), final_model_path)
with open("./uniformer_Wmodels/2cams/training_history.pkl", "wb") as f:
    pickle.dump(history, f)

elapsed = time.time() - start_time
print(f"Final saved to {final_model_path} | Time: {elapsed:.2f}s")
wandb.finish()
