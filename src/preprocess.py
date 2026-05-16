"""
Extract individual panels from Manga109-s raw pages.

Reads frame bounding boxes from XML annotations, crops panels,
filters by text coverage and minimum size, resizes to 768x768 RGB PNG.

Usage:
    python src/preprocess.py                       # full run (all panels)
    python src/preprocess.py --dry-run             # 3 works only
    python src/preprocess.py --panels-per-work 4   # sample 4 panels per work
"""

import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from utils import compute_text_coverage, parse_manga109_page, resize_pad_to_square, setup_logging

DATA_ROOT = Path("data/raw/Manga109s_released_2023_12_07")
ANN_DIR = DATA_ROOT / "annotations.v2020.12.18"
IMG_DIR = DATA_ROOT / "images"
BOOKS_FILE = DATA_ROOT / "books.txt"
OUT_DIR = Path("data/processed")
SKIP_LOG = OUT_DIR / "skip_log.txt"

MIN_SIDE = 150        # pixels — panels smaller than this are skipped
TEXT_THRESH = 0.40    # skip panel if text covers more than this fraction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Process first 3 works only")
    p.add_argument("--panels-per-work", type=int, default=0,
                   help="Sample at most N panels per work (0 = all panels)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    return p.parse_args()


def load_books(data_root: Path) -> list:
    books_path = data_root / "books.txt"
    return [l.strip() for l in books_path.read_text().splitlines() if l.strip()]


def extract_panels_from_work(work: str, data_root: Path, out_dir: Path,
                             skip_fh, logger, panels_per_work: int = 0) -> int:
    """Extract valid panels from one manga work. Returns count saved.

    If panels_per_work > 0, randomly samples that many pages and takes the
    first valid panel per page, giving diversity across works.
    """
    ann_path = data_root / "annotations.v2020.12.18" / f"{work}.xml"
    img_base = data_root / "images" / work

    if not ann_path.exists():
        logger.warning(f"No annotation XML for {work}, skipping")
        return 0

    try:
        tree = ET.parse(ann_path)
    except ET.ParseError as e:
        logger.error(f"XML parse error for {work}: {e}")
        return 0

    pages = tree.findall(".//page")
    pages_with_frames = [p for p in pages if p.findall("frame")]

    if panels_per_work > 0:
        # Sample pages spread across the work for visual diversity
        sampled = random.sample(pages_with_frames,
                                min(panels_per_work, len(pages_with_frames)))
    else:
        sampled = pages_with_frames

    saved = 0
    for page_elem in sampled:
        page_data = parse_manga109_page(page_elem)
        page_idx = page_data["index"]

        img_path = img_base / f"{page_idx:03d}.jpg"
        if not img_path.exists():
            skip_fh.write(f"{work} page {page_idx:03d}: image file missing\n")
            continue

        try:
            page_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            skip_fh.write(f"{work} page {page_idx:03d}: cannot open image ({e})\n")
            continue

        # When sampling, take first valid panel per page; otherwise take all
        found_one = False
        for panel_idx, frame in enumerate(page_data["frames"]):
            if panels_per_work > 0 and found_one:
                break

            fx1, fy1, fx2, fy2 = frame
            w, h = fx2 - fx1, fy2 - fy1
            if w < MIN_SIDE or h < MIN_SIDE:
                skip_fh.write(
                    f"{work} p{page_idx:03d} f{panel_idx:02d}: too small ({w}x{h})\n")
                continue

            coverage = compute_text_coverage(frame, page_data["texts"])
            if coverage > TEXT_THRESH:
                skip_fh.write(
                    f"{work} p{page_idx:03d} f{panel_idx:02d}: text coverage {coverage:.2f}\n")
                continue

            cropped = page_img.crop((fx1, fy1, fx2, fy2))
            panel = resize_pad_to_square(cropped, size=768)

            stem = f"{work}_{page_idx:03d}_{panel_idx:02d}"
            out_path = out_dir / f"{stem}.png"
            panel.save(out_path, "PNG")
            saved += 1
            found_one = True

    return saved


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    logger = setup_logging("preprocess")
    books = load_books(data_root)

    if args.dry_run:
        books = books[:3]
        logger.info(f"DRY RUN — processing {len(books)} works: {books}")
    else:
        logger.info(f"Processing {len(books)} works")

    if args.panels_per_work:
        logger.info(f"Sampling {args.panels_per_work} panels per work")

    total_saved = 0

    skip_path = out_dir / ("skip_log_dryrun.txt" if args.dry_run else "skip_log.txt")
    with open(skip_path, "w") as skip_fh:
        for work in tqdm(books, desc="Works"):
            saved = extract_panels_from_work(
                work, data_root, out_dir, skip_fh, logger,
                panels_per_work=args.panels_per_work)
            total_saved += saved
            logger.info(f"{work}: {saved} panels saved")

    logger.info(f"Done. Total panels saved: {total_saved}")
    logger.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
