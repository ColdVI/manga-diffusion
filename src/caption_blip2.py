"""
Generate BLIP-2 auto-captions for processed panels (Condition A).

Loads Salesforce/blip2-opt-2.7b and captions every image in --img-dir,
saving one .txt file per image to data/captions/blip2/.

Usage:
    python src/caption_blip2.py              # caption data/processed/
    python src/caption_blip2.py --dry-run    # first 3 images only
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from utils import setup_logging

IMG_DIR = Path("data/processed")
OUT_DIR = Path("data/captions/blip2")
MANIFEST = Path("data/split_manifest.json")
MODEL_ID = "Salesforce/blip2-opt-2.7b"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", default=str(IMG_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--manifest", default=str(MANIFEST),
                   help="If given, caption only train panels from manifest")
    p.add_argument("--dry-run", action="store_true",
                   help="Process first 3 images only")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip images that already have a caption file")
    return p.parse_args()


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device: str, logger):
    logger.info(f"Loading {MODEL_ID} on {device} ...")
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    processor = Blip2Processor.from_pretrained(MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map=None)
    model = model.to(device)
    model.eval()
    logger.info("Model loaded")
    return processor, model


def caption_image(img_path: Path, processor, model, device: str) -> str:
    """Return a single caption string for the given image."""
    dtype = next(model.parameters()).dtype
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device, dtype)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=50)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def main():
    args = parse_args()
    logger = setup_logging("caption_blip2")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which images to caption
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
        logger.info(f"Skipping {before - len(images)} already-captioned images")

    if not images:
        logger.info("Nothing to do.")
        return

    device = pick_device()
    processor, model = load_model(device, logger)

    errors = 0
    for img_path in tqdm(images, desc="BLIP-2"):
        try:
            caption = caption_image(img_path, processor, model, device)
            (out_dir / f"{img_path.stem}.txt").write_text(caption)
        except Exception as e:
            logger.error(f"Failed {img_path.name}: {e}")
            errors += 1

    logger.info(f"Done. {len(images) - errors} captions written, {errors} errors.")
    logger.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
