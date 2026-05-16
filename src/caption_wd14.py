"""
Generate WD14 tagger captions for processed panels (Condition B).

Downloads SmilingWolf/wd-v1-4-convnext-tagger-v2 ONNX model from HuggingFace
and produces Danbooru-style tag strings for each image.

Usage:
    python src/caption_wd14.py              # caption data/processed/
    python src/caption_wd14.py --dry-run    # first 3 images only
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import onnxruntime as rt
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

from utils import setup_logging

IMG_DIR = Path("data/processed")
OUT_DIR = Path("data/captions/wd14")
MANIFEST = Path("data/split_manifest.json")
MODEL_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
MODEL_FILE = "model.onnx"
TAGS_FILE = "selected_tags.csv"
THRESHOLD = 0.35
INPUT_SIZE = 448          # WD14 convnext expects 448×448
# category 9 = copyright tags (noisy/irrelevant for style training)
SKIP_CATEGORIES = {9}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", default=str(IMG_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--manifest", default=str(MANIFEST))
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--dry-run", action="store_true",
                   help="Process first 3 images only")
    p.add_argument("--skip-existing", action="store_true", default=True)
    return p.parse_args()


def load_model_and_tags(logger):
    logger.info(f"Downloading {MODEL_REPO} ONNX model ...")
    model_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
    tags_path = hf_hub_download(MODEL_REPO, TAGS_FILE)

    sess = rt.InferenceSession(model_path,
                                providers=["CPUExecutionProvider"])
    logger.info("ONNX session created")

    tags = []
    with open(tags_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tags.append({"name": row["name"], "category": int(row["category"])})

    logger.info(f"Loaded {len(tags)} tags")
    return sess, tags


def preprocess(img_path: Path, size: int = INPUT_SIZE) -> np.ndarray:
    """Resize image to size×size and convert to float32 BGR array."""
    img = Image.open(img_path).convert("RGB")
    # Pad to square first
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    canvas = canvas.resize((size, size), Image.LANCZOS)

    arr = np.array(canvas, dtype=np.float32)
    arr = arr[:, :, ::-1]   # RGB → BGR (WD14 convention)
    return np.expand_dims(arr, 0)   # (1, H, W, C)


def tag_image(img_path: Path, sess, tags: list, threshold: float) -> str:
    """Return comma-separated tag string above threshold."""
    arr = preprocess(img_path)
    input_name = sess.get_inputs()[0].name
    probs = sess.run(None, {input_name: arr})[0][0]   # shape: (num_tags,)

    results = []
    for prob, tag in zip(probs, tags):
        if tag["category"] in SKIP_CATEGORIES:
            continue
        if float(prob) >= threshold:
            results.append((float(prob), tag["name"].replace("_", " ")))

    results.sort(reverse=True)
    return ", ".join(name for _, name in results)


def main():
    args = parse_args()
    logger = setup_logging("caption_wd14")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if Path(args.manifest).exists():
        with open(args.manifest) as f:
            manifest = json.load(f)
        images = [Path(p) for p in manifest["train_panels"]]
        logger.info(f"Captioning {len(images)} train panels from manifest")
    else:
        images = sorted(Path(args.img_dir).glob("*.png"))
        logger.info(f"Captioning all {len(images)} images in {args.img_dir}")

    if args.dry_run:
        images = images[:3]
        logger.info(f"DRY RUN — processing {len(images)} images")

    if args.skip_existing:
        before = len(images)
        images = [p for p in images if not (out_dir / f"{p.stem}.txt").exists()]
        logger.info(f"Skipping {before - len(images)} already-tagged images")

    if not images:
        logger.info("Nothing to do.")
        return

    sess, tags = load_model_and_tags(logger)

    errors = 0
    for img_path in tqdm(images, desc="WD14"):
        try:
            caption = tag_image(img_path, sess, tags, args.threshold)
            (out_dir / f"{img_path.stem}.txt").write_text(caption)
        except Exception as e:
            logger.error(f"Failed {img_path.name}: {e}")
            errors += 1

    logger.info(f"Done. {len(images) - errors} tags written, {errors} errors.")
    logger.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
