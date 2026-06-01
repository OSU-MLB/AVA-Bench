#!/usr/bin/env python3
"""Download an AVA-Bench capability from HuggingFace and materialize it into the
local layout the training code expects:

    <out_root>/<Cap>/train.json          # list of {id, image, conversations}
    <out_root>/<Cap>/images/<id>.<ext>   # decoded images

The `image` field written into train.json is "<Cap>/images/<id>.<ext>", so the
loader resolves it with  image_folder = <out_root>  (== AVA-Bench/train).

HF source: act13/AVA-Bench  (parquet, one config per capability, embedded PIL
images + `conversations`).

Usage:
    python prepare_data.py --cap Counting --out-root /path/AVA-Bench/train
    python prepare_data.py --all          --out-root /path/AVA-Bench/train
"""
import argparse
import json
import os

# HF config name (== capability dir) for every AVA capability in act13/AVA-Bench
CAPABILITIES = [
    "Absolute_depth", "Action", "Color", "Counting", "Emotion", "Fine-grained",
    "Localization", "OCR", "Orientation", "Recognition", "Relative_depth",
    "Scene_Classification", "Spatial", "Texture",
]

HF_REPO = "act13/AVA-Bench"


def _ext_for(img):
    fmt = (getattr(img, "format", None) or "JPEG").upper()
    return {"JPEG": "jpg", "MPO": "jpg"}.get(fmt, fmt.lower())


def prepare(cap, out_root, force=False):
    from datasets import load_dataset

    cap_dir = os.path.join(out_root, cap)
    img_dir = os.path.join(cap_dir, "images")
    json_path = os.path.join(cap_dir, "train.json")

    if os.path.exists(json_path) and not force:
        print(f"[skip] {cap}: {json_path} already exists (use --force to rebuild)")
        return

    os.makedirs(img_dir, exist_ok=True)
    print(f"[load] {cap}: downloading from {HF_REPO} ...")
    ds = load_dataset(HF_REPO, name=cap, split="train")

    records = []
    for i, ex in enumerate(ds):
        rid = ex.get("id")
        if rid is None or rid == "":
            rid = f"{cap}_{i:08d}"
        rid = str(rid).replace("/", "_")

        img = ex["image"]
        ext = _ext_for(img)
        fname = f"{rid}.{ext}"
        fpath = os.path.join(img_dir, fname)
        if not os.path.exists(fpath):
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(fpath)

        rec = {k: v for k, v in ex.items() if k != "image"}
        rec["id"] = rid
        rec["image"] = f"{cap}/images/{fname}"
        records.append(rec)

        if (i + 1) % 2000 == 0:
            print(f"       {cap}: {i + 1} rows...")

    tmp = json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(records, f)
    os.replace(tmp, json_path)
    print(f"[done] {cap}: {len(records)} rows -> {json_path}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cap", help="single capability dir name, e.g. Counting")
    g.add_argument("--all", action="store_true", help="prepare every capability")
    ap.add_argument("--out-root", required=True,
                    help="destination root (AVA-Bench/train)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if train.json already exists")
    args = ap.parse_args()

    caps = CAPABILITIES if args.all else [args.cap]
    for cap in caps:
        if cap not in CAPABILITIES:
            raise SystemExit(f"Unknown capability '{cap}'. "
                             f"Valid: {', '.join(CAPABILITIES)}")
        prepare(cap, args.out_root, force=args.force)


if __name__ == "__main__":
    main()
