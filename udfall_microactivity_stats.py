# udfall_microactivity_stats.py
import os, json, csv, math, re
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# --------------------
# 1) Settings
# --------------------
FPS = 25  # effective label rate (Hz)
ROOTS = [
    Path("./micro_activity_output_cleaned"),  # main dataset root
    Path("./"),                       # current folder (if samples exist here)
    Path("/mnt/data"),                # optional debug path
]
OUTPUT_DIR = Path("./udfall_stats-cutrepo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep only existing roots
ROOTS = [r for r in ROOTS if r.exists()]

# --------------------
# 2) Official Micro -> Macro mapping (from thesis Table 4.3)
# Labels are normalized (lowercase, '_' -> ' ', collapse spaces) before lookup.
OFFICIAL_MAP = {
    "fall lateral": "Fall",
    "fall frontal": "Fall",
    "fall crouch": "Fall",
    "fall rolling": "Fall",
    "sit up from lying": "ADL",
    "stand still": "ADL",
    "lie down from sitting": "ADL",
    "sit down from standing": "ADL",
    "stand up from floor": "ADL",
    "rolling bed": "ADL",
    "sit still": "ADL",
    "stand up from sit": "ADL",
    "walking": "ADL",
    "pick up object": "ADL",
    "lie down on the floor": "ADL",
    "lie still": "Lying",
}

def _normalize_label(s: str) -> str:
    """Lowercase, replace underscores with spaces, and collapse multiple spaces."""
    s = s.lower().replace("_", " ").strip()
    return re.sub(r"\s+", " ", s)

def to_macro(label: str) -> str:
    """Map a micro-activity to its macro group using the official table."""
    return OFFICIAL_MAP.get(_normalize_label(label), "ADL")

# --------------------
# 3) Discover annotation files (ONLY top-level scenario folders)
# Pattern: <root>/*/ann/per_frame_micro_activities.json
# --------------------
json_files = []
for root in ROOTS:
    json_files.extend(root.glob("*/ann/per_frame_micro_activities.json"))

if not json_files:
    print("No per_frame_micro_activities.json found under:", ROOTS)
    raise SystemExit(0)

print(f"Found {len(json_files)} annotation files.")

# --------------------
# 4) Read labels and count occurrences
# --------------------
global_counts = Counter()     # micro-activity -> total frames
macro_counts  = Counter()     # macro-activity  -> total frames
scenario_activity_counts = [] # (scenario_path, Counter(micro-activity))

def load_labels(path: Path):
    """Return per-frame labels as a list (supports list or dict JSON)."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Keys may be strings; try numeric sort, else keep insertion order.
        try:
            items = sorted(((int(k), v) for k, v in data.items()), key=lambda x: x[0])
        except Exception:
            items = list(data.items())
        return [v for _, v in items]
    raise ValueError(f"Unexpected JSON at {path}: {type(data)}")

for jf in json_files:
    try:
        labels = load_labels(jf)
    except Exception as e:
        print("Skip (read error):", jf, e)
        continue

    c = Counter(labels)
    scenario_name = str(jf.parent.parent)  # <root>/<SCENARIO>/ann
    scenario_activity_counts.append((scenario_name, c))

    global_counts.update(c)

# --------------------
# 5) Global stats + CSV exports
# --------------------
total_frames  = sum(global_counts.values())
total_seconds = total_frames / FPS

# CSV: micro-activity counts
csv_path = OUTPUT_DIR / "activity_counts.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["micro_activity", "frame_count", "seconds", "percent"])
    for act, cnt in global_counts.most_common():
        w.writerow([act, cnt, f"{cnt/FPS:.2f}", f"{(cnt/total_frames*100 if total_frames else 0):.2f}"])

# CSV: scenario × activity (wide format) for heatmaps
all_acts = sorted(global_counts.keys())
heat_csv = OUTPUT_DIR / "scenario_activity_heatmap.csv"
with open(heat_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["scenario"] + all_acts + ["total_frames"])
    for scen_name, cnts in scenario_activity_counts:
        tf = sum(cnts.values())
        row = [scen_name] + [cnts.get(a, 0) for a in all_acts] + [tf]
        w.writerow(row)

# --------------------
# 6) Macro-activity share (using official mapping)
# --------------------
for micro, cnt in global_counts.items():
    macro_counts[to_macro(micro)] += cnt

# --------------------
# 7) Plots (PNG)
# --------------------
# 7.1 Bar chart: micro-activity frequency
plt.figure(figsize=(12, 6))
acts, counts = zip(*global_counts.most_common()) if global_counts else ([], [])
plt.bar(range(len(acts)), counts)
plt.xticks(range(len(acts)), acts, rotation=75, ha="right")
plt.ylabel("Frame count")
plt.title("Micro-activity distribution (frames)")
plt.tight_layout()
bar_path = OUTPUT_DIR / "microactivity_counts.png"
plt.savefig(bar_path, dpi=200)
plt.close()

# 7.2 Pie chart: macro-activity share
if macro_counts:
    plt.figure(figsize=(6, 6))
    labels = list(macro_counts.keys())
    sizes  = [macro_counts[k] for k in labels]
    plt.pie(sizes, labels=labels, autopct=lambda p: f"{p:.1f}%" if p > 0 else "")
    plt.title("Macro-activity share (by time)")
    pie_path = OUTPUT_DIR / "macroactivity_share.png"
    plt.savefig(pie_path, dpi=200)
    plt.close()

# 7.3 Console summary
print(f"Total frames: {total_frames:,}  (~{total_seconds/60:.1f} minutes at {FPS} Hz)")
print("Top-10 micro-activities:")
for i, (a, c) in enumerate(global_counts.most_common(10), 1):
    print(f"{i:>2}. {a:35s} {c:>10,d} frames  ({c/FPS:.1f}s)")

print("\nMacro-activity share (frames):", dict(macro_counts))
print("\nSaved:")
print("  -", csv_path)
print("  -", heat_csv)
print("  -", bar_path)
print("  -", pie_path)
