"""
Build train/test split manifest from processed panels.

Splits by work (not by page) to prevent data leakage.
Writes data/split_manifest.json and optionally assembles data/train_set/.

Usage:
    # Step 1: create the split manifest
    python src/build_dataset.py --split

    # Step 2 (after captioning): assemble train_set/ folders for kohya
    python src/build_dataset.py --assemble

    python src/build_dataset.py --dry-run --split   # test only
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from utils import setup_logging

PROCESSED_DIR = Path("data/processed")
TRAIN_SET_DIR = Path("data/train_set")
CAPTION_DIRS = {
    "blip2": Path("data/captions/blip2"),
    "wd14": Path("data/captions/wd14"),
}
MANIFEST_PATH = Path("data/split_manifest.json")
KOHYA_REPEAT = 10      # folder prefix: 10_gknoir → repeat dataset 10× per epoch
CONCEPT = "gknoir"
TRIGGER = "gknoir style"  # prepended to every caption so LoRA binds the token

TARGET_TRAIN = 200
TARGET_TEST = 40


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", action="store_true",
                   help="Create train/test split manifest")
    p.add_argument("--assemble", action="store_true",
                   help="Assemble train_set/ from manifest + caption files")
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions without writing files")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    p.add_argument("--manifest", default=str(MANIFEST_PATH))
    return p.parse_args()


def gather_panels_by_work(processed_dir: Path) -> dict:
    """Return {work_name: [panel_path, ...]} for all processed panels."""
    by_work = defaultdict(list)
    for p in sorted(processed_dir.glob("*.png")):
        # Filename: {work}_{page:03d}_{panel:02d}.png
        # Work name may contain underscores, so split from right
        parts = p.stem.rsplit("_", 2)
        if len(parts) == 3:
            work = parts[0]
            by_work[work].append(p)
    return dict(by_work)


def split_works(works: list, train_ratio: float, seed: int) -> tuple:
    """Split work list into train/test by work (not by panel)."""
    rng = random.Random(seed)
    shuffled = works.copy()
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_ratio))
    return shuffled[:n_train], shuffled[n_train:]


def sample_panels(panels: list, target: int, seed: int) -> list:
    """Randomly sample up to target panels, preserving work diversity."""
    rng = random.Random(seed)
    if len(panels) <= target:
        return panels
    return rng.sample(panels, target)


def create_manifest(args, logger):
    """Create and save the train/test split manifest."""
    processed_dir = Path(args.processed_dir)
    by_work = gather_panels_by_work(processed_dir)

    if not by_work:
        logger.error(f"No panels found in {processed_dir}. Run preprocess.py first.")
        return

    works = sorted(by_work.keys())
    logger.info(f"Found {len(works)} works, {sum(len(v) for v in by_work.values())} panels")

    train_works, test_works = split_works(works, args.train_ratio, args.seed)
    logger.info(f"Train works: {len(train_works)}, Test works: {len(test_works)}")

    train_panels_all = [str(p) for w in train_works for p in by_work[w]]
    test_panels_all = [str(p) for w in test_works for p in by_work[w]]

    rng = random.Random(args.seed)
    train_panels = sorted(rng.sample(train_panels_all,
                                     min(TARGET_TRAIN, len(train_panels_all))))
    test_panels = sorted(rng.sample(test_panels_all,
                                    min(TARGET_TEST, len(test_panels_all))))

    logger.info(f"Train panels: {len(train_panels)}, Test panels: {len(test_panels)}")

    manifest = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "train_works": train_works,
        "test_works": test_works,
        "train_panels": train_panels,
        "test_panels": test_panels,
    }

    if args.dry_run:
        logger.info(f"DRY RUN — would write manifest to {args.manifest}")
        logger.info(f"Train sample: {train_panels[:3]}")
        logger.info(f"Test sample:  {test_panels[:3]}")
        return

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {manifest_path}")

    # Copy test panels to data/test_set/ (frozen reference set)
    test_set_dir = Path("data/test_set")
    test_set_dir.mkdir(parents=True, exist_ok=True)
    for p in test_panels:
        shutil.copy2(p, test_set_dir / Path(p).name)
    logger.info(f"Test panels copied to {test_set_dir}/")


def assemble_train_set(args, logger):
    """Assemble kohya-ready train_set/ from manifest + caption .txt files."""
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(f"Manifest not found at {manifest_path}. Run --split first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    train_panels = manifest["train_panels"]
    logger.info(f"Assembling train_set/ for {len(train_panels)} panels")

    for condition, cap_dir in CAPTION_DIRS.items():
        folder_name = f"{KOHYA_REPEAT}_{CONCEPT}"
        dest = TRAIN_SET_DIR / condition / folder_name
        if not args.dry_run:
            dest.mkdir(parents=True, exist_ok=True)

        missing_caps = 0
        copied = 0
        for panel_path_str in train_panels:
            panel_path = Path(panel_path_str)
            stem = panel_path.stem
            cap_path = cap_dir / f"{stem}.txt"

            if not cap_path.exists():
                missing_caps += 1
                continue

            if not args.dry_run:
                shutil.copy2(panel_path, dest / panel_path.name)
                # Prepend trigger word so the LoRA binds gknoir to this style
                raw = cap_path.read_text().strip()
                final = f"{TRIGGER}, {raw}" if not raw.startswith(TRIGGER) else raw
                (dest / f"{stem}.txt").write_text(final)
            copied += 1

        if missing_caps:
            logger.warning(f"[{condition}] {missing_caps} panels missing captions — "
                           f"run caption_{condition}.py first")
        logger.info(f"[{condition}] {'Would copy' if args.dry_run else 'Copied'} "
                    f"{copied} image+caption pairs → {dest}")


def main():
    args = parse_args()
    logger = setup_logging("build_dataset")

    if not args.split and not args.assemble:
        logger.error("Specify --split and/or --assemble")
        return

    if args.split:
        create_manifest(args, logger)
    if args.assemble:
        assemble_train_set(args, logger)


if __name__ == "__main__":
    main()
