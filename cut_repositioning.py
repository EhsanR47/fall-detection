import os, json, re, shutil, glob
from pathlib import Path

# ---------- Path Configuration ----------
ROOT_FRAMES = Path("./cam2-resized_frames")         # Directory containing all extracted frames
ROOT_ANN    = Path("./micro_activity_output")       # Directory containing micro-activity JSON files
OUT_ROOT    = Path("./cam2-cleaned_frames")         # Output directory for cleaned frames (without repositioning)

# If you want to process only specific activities, list them here. Otherwise, set to None to process all.
ONLY_ACTIVITIES = None  # Example: ["Actor_1_Bed", "Actor_1_Chair"]

# Instead of copying files, create hardlinks (saves disk space). 
# If your filesystem does not support hardlinks, set this to False.
USE_HARDLINKS = True

# ---------- Helper Functions ----------
def is_repositioning(label: str) -> bool:
    """Return True if the label indicates any repositioning activity."""
    return "repositioning" in label.lower()

def load_labels(json_path: Path) -> dict[int, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Convert keys (which may be strings) to integers
    return {int(k): v for k, v in raw.items()}

def ensure_dir(p: Path):
    """Create the directory (and its parents) if it doesn't already exist."""
    p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path, use_hardlinks: bool = True):
    """Create a hardlink or fall back to copying the file if hardlink fails."""
    if use_hardlinks:
        try:
            if dst.exists():
                return
            os.link(src, dst)  # Create a hardlink
            return
        except Exception:
            pass
    # Fallback to copying
    if not dst.exists():
        shutil.copy2(src, dst)

def collect_subfolders(activity_dir: Path):
    """
    Based on your folder structure, there may be subfolders like "Actor_1_Bed CAM 2".
    If not, return the main directory itself.
    """
    subs = [p for p in activity_dir.iterdir() if p.is_dir()]
    return subs if subs else [activity_dir]

def clean_activity(activity_name: str):
    """Process one activity: remove repositioning frames and copy/ link valid ones to the output."""
    activity_frames_dir = ROOT_FRAMES / activity_name
    ann_json = ROOT_ANN / activity_name / "ann" / "per_frame_micro_activities.json"

    if not activity_frames_dir.exists():
        print(f"[SKIP] frames dir missing: {activity_frames_dir}")
        return

    if not ann_json.exists():
        print(f"[SKIP] ann json missing: {ann_json}")
        return

    labels = load_labels(ann_json)
    keep_indices = sorted([i for i, lab in labels.items() if not is_repositioning(lab)])
    drop_indices = sorted([i for i, lab in labels.items() if is_repositioning(lab)])

    subfolders = collect_subfolders(activity_frames_dir)

    kept, dropped, missing = 0, 0, 0

    for sub in subfolders:
        out_sub = OUT_ROOT / activity_name / sub.name
        ensure_dir(out_sub)

        for idx in keep_indices:
            # Frame naming convention based on your dataset: frame_<index>.jpg
            fname = f"frame_{idx}.jpg"
            src = sub / fname
            dst = out_sub / fname
            if src.exists():
                link_or_copy(src, dst, USE_HARDLINKS)
                kept += 1
            else:
                missing += 1

        # Uncomment this block if you also want to save repositioning frames (for inspection)
        # bad_sub = OUT_ROOT / "_dropped" / activity_name / sub.name
        # ensure_dir(bad_sub)
        # for idx in drop_indices:
        #     fname = f"frame_{idx}.jpg"
        #     src = sub / fname
        #     dst = bad_sub / fname
        #     if src.exists():
        #         link_or_copy(src, dst, USE_HARDLINKS)
        #         dropped += 1

    print(f"[DONE] {activity_name}: kept={kept}, dropped_labels={len(drop_indices)}, missing_files={missing}")

def main():
    """Main execution function that iterates over all activities and cleans them."""
    activities = [p.name for p in ROOT_FRAMES.iterdir() if p.is_dir()]
    if ONLY_ACTIVITIES:
        activities = [a for a in activities if a in ONLY_ACTIVITIES]

    ensure_dir(OUT_ROOT)

    for act in sorted(activities):
        clean_activity(act)

if __name__ == "__main__":
    main()
