"""
Compute per-channel normalization stats (mean & std in [0,1]) for your dataset.

- Loads the same pickle list you use: [(frame_paths:list[str], label:int), ...]
- Does the same actor-based split and uses ONLY the TRAIN split
- Uniformly samples a subset of frames (e.g., 10k) for speed
- Accumulates sum and sum of squares to get exact mean/std

Paste the printed mean/std into your HP["normalize_mean"], HP["normalize_std"].
"""

import os, pickle, random, time, json
from collections import defaultdict
from PIL import Image
import numpy as np
from tqdm import tqdm

# --------- Config ---------
PICKLE_PATH      = "./f_seq_stride8.pkl"  # your dataset pickle
SEQ_LEN_USED     = 16                     # not used for stats; we sample frames globally
SAMPLE_FRAMES    = 40000                  # how many frames to sample from TRAIN (increase if fast)
RANDOM_SEED      = 42                     # reproducibility
SAVE_JSON        = "./channel_stats.json" # optional output file (set to None to skip saving)
# --------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---- Load dataset [(frame_paths, label), ...] ----
with open(PICKLE_PATH, "rb") as f:
    dataset = pickle.load(f)

print(f"Loaded dataset with {len(dataset)} sequences.")

# ---- Actor-based split (same logic as your training) ----
actor_to_samples = defaultdict(list)
for seq_paths, label in dataset:
    parts = seq_paths[0].split(os.sep)
    actor_folder = next(p for p in parts if p.startswith("Actor_"))
    actor_id = "_".join(actor_folder.split("_")[:2])
    actor_to_samples[actor_id].append((seq_paths, label))

all_actors = sorted(actor_to_samples.keys())
rng = random.Random(RANDOM_SEED)
rng.shuffle(all_actors)

split_idx = int(0.8 * len(all_actors))
train_actors, val_actors = all_actors[:split_idx], all_actors[split_idx:]

train_samples = [s for a in train_actors for s in actor_to_samples[a]]
print(f"Train sequences: {len(train_samples)} | Val sequences: {len(dataset) - len(train_samples)}")

# ---- Collect ALL frame paths from TRAIN split ----
train_frame_paths = []
for frame_paths, _ in train_samples:
    train_frame_paths.extend(frame_paths)

n_total_frames = len(train_frame_paths)
print(f"Total train frames available: {n_total_frames}")

# ---- Decide how many to sample ----
if SAMPLE_FRAMES is None or SAMPLE_FRAMES <= 0 or SAMPLE_FRAMES > n_total_frames:
    sampled_paths = train_frame_paths
else:
    sampled_paths = rng.sample(train_frame_paths, SAMPLE_FRAMES)

print(f"Sampling {len(sampled_paths)} frames to estimate stats.")

# ---- Accumulate sums and sums of squares per channel ----
# We compute over pixel values scaled to [0,1].
sum_c     = np.zeros(3, dtype=np.float64)
sumsq_c   = np.zeros(3, dtype=np.float64)
count_pix = 0

t0 = time.time()
for p in tqdm(sampled_paths, desc="Computing mean/std"):
    img = Image.open(p).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H x W x 3, in [0,1]
    # Sum over H,W for each channel
    sum_c   += arr.reshape(-1, 3).sum(axis=0)
    sumsq_c += (arr.reshape(-1, 3) ** 2).sum(axis=0)
    count_pix += arr.shape[0] * arr.shape[1]

# E[x] and E[x^2]
mean = sum_c / count_pix
var  = sumsq_c / count_pix - mean**2
std  = np.sqrt(np.clip(var, a_min=0.0, a_max=None))

elapsed = time.time() - t0

mean_list = [float(round(x, 6)) for x in mean]
std_list  = [float(round(x, 6)) for x in std]

print("\n==== Channel-wise stats (use in Normalize) ====")
print(f"mean = {mean_list}")
print(f"std  = {std_list}")
print(f"Time: {elapsed:.2f}s for {len(sampled_paths)} frames")

# ---- Optionally save to JSON ----
if SAVE_JSON:
    out = {
        "mean": mean_list,
        "std": std_list,
        "sampled_frames": len(sampled_paths),
        "total_train_frames": n_total_frames,
        "seed": RANDOM_SEED,
        "pickle": PICKLE_PATH,
    }
    with open(SAVE_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved stats to {SAVE_JSON}")
