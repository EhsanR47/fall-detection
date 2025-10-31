import json
from pathlib import Path

# ---------- Paths ----------
ROOT_ANN_IN  = Path("./micro_activity_output")          # Original annotation directory
ROOT_ANN_OUT = Path("./micro_activity_output_cleaned")  # Output directory for cleaned annotations

# If you want to clean only specific activities, list them here; otherwise set to None to process all
ONLY_ACTIVITIES = None  # Example: ["Actor_1_Bed", "Actor_1_Chair"]

def is_repositioning(label: str) -> bool:
    """Return True if the label indicates any repositioning activity (case-insensitive)."""
    return "repositioning" in label.lower()

def load_labels_with_meta(json_path: Path):
    """
    Load labels and detect whether the file uses the 'root' key structure.
    Returns:
        labels (dict[int, str]): Mapping from frame index to label
        had_root (bool): True if the input JSON had a top-level 'root' key
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "root" in raw and isinstance(raw["root"], dict):
        d = raw["root"]
        had_root = True
    else:
        d = raw
        had_root = False

    labels = {int(k): v for k, v in d.items()}
    return labels, had_root

def save_labels_with_meta(json_path_out: Path, labels_dict_int: dict[int, str], had_root: bool):
    """
    Save cleaned labels using the same format as the input:
    - If the input had a 'root' key, the output will also include 'root'.
    - Keys are written as strings and sorted numerically.
    """
    json_path_out.parent.mkdir(parents=True, exist_ok=True)
    items_sorted = sorted(labels_dict_int.items(), key=lambda kv: kv[0])
    as_str_keys = {str(k): v for k, v in items_sorted}
    to_write = {"root": as_str_keys} if had_root else as_str_keys

    with open(json_path_out, "w", encoding="utf-8") as f:
        json.dump(to_write, f, ensure_ascii=False, indent=2, sort_keys=True)

def clean_one_activity(activity_name: str):
    """Clean one activity by removing all labels containing 'repositioning'."""
    src_json = ROOT_ANN_IN / activity_name / "ann" / "per_frame_micro_activities.json"
    if not src_json.exists():
        print(f"[SKIP] Annotation JSON missing: {src_json}")
        return

    labels, had_root = load_labels_with_meta(src_json)

    # Remove all labels that contain 'repositioning'
    cleaned = {idx: lab for idx, lab in labels.items() if not is_repositioning(lab)}

    dst_json = ROOT_ANN_OUT / activity_name / "ann" / "per_frame_micro_activities.json"
    save_labels_with_meta(dst_json, cleaned, had_root)

    print(f"[DONE] {activity_name}: in={len(labels)} -> out={len(cleaned)} (removed={len(labels)-len(cleaned)})")
    print(f"       wrote: {dst_json}")

def main():
    """Iterate through all activities and clean their annotation files."""
    activities = [p.name for p in ROOT_ANN_IN.iterdir() if p.is_dir()]
    if ONLY_ACTIVITIES:
        activities = [a for a in activities if a in ONLY_ACTIVITIES]

    for act in sorted(activities):
        clean_one_activity(act)

if __name__ == "__main__":
    main()
