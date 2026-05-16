"""
Compute DINOv2 style similarity between generated images and test set.

Extracts patch-level features from all generated images and all test images,
then computes the mean pairwise cosine similarity between the two sets.

Usage:
    python evaluation/compute_dino.py --condition blip2
    python evaluation/compute_dino.py --all
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import setup_logging

TEST_SET_DIR = Path("data/test_set")
RESULTS_DIR = Path("evaluation/results")
METRICS_CSV = RESULTS_DIR / "metrics_summary.csv"
CONDITIONS = ["baseline", "blip2", "wd14"]


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--condition", choices=CONDITIONS)
    g.add_argument("--all", action="store_true")
    return p.parse_args()


def extract_features(image_paths: list, model, processor, device: str, desc: str) -> torch.Tensor:
    """Return (N, D) CLS token feature matrix for a list of image paths."""
    feats = []
    for img_path in tqdm(image_paths, desc=desc):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        # CLS token from last hidden state
        cls = out.last_hidden_state[:, 0, :]  # (1, D)
        feats.append(cls.cpu())
    return torch.cat(feats, dim=0)  # (N, D)


def compute_dino_for(condition: str, logger) -> float:
    gen_dir = RESULTS_DIR / condition / "generated"
    gen_images = sorted(gen_dir.glob("*.png"))
    test_images = sorted(TEST_SET_DIR.glob("*.png"))

    if not gen_images:
        logger.error(f"No generated images at {gen_dir}")
        return float("nan")
    if not test_images:
        logger.error(f"No test images at {TEST_SET_DIR}")
        return float("nan")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[{condition}] Loading DINOv2 on {device} ...")
    model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
    processor = AutoProcessor.from_pretrained("facebook/dinov2-large")
    model.eval()

    gen_feats = extract_features(gen_images, model, processor, device,
                                  f"DINO gen [{condition}]")
    test_feats = extract_features(test_images, model, processor, device,
                                   "DINO test")

    gen_norm = F.normalize(gen_feats, dim=1)
    test_norm = F.normalize(test_feats, dim=1)
    # Mean pairwise cosine similarity between the two sets
    sim_matrix = gen_norm @ test_norm.T  # (N_gen, N_test)
    mean_sim = sim_matrix.mean().item()
    logger.info(f"[{condition}] DINOv2 style similarity = {mean_sim:.4f}")
    return mean_sim


def update_csv(condition: str, dino_sim: float, logger):
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    if METRICS_CSV.exists():
        with open(METRICS_CSV) as f:
            for row in csv.DictReader(f):
                rows[row["run_name"]] = row

    if condition not in rows:
        rows[condition] = {"run_name": condition, "fid": "", "clip_score": "", "dino_sim": "", "notes": ""}
    rows[condition]["dino_sim"] = f"{dino_sim:.4f}"

    with open(METRICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "fid", "clip_score", "dino_sim", "notes"])
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)
    logger.info(f"CSV updated: {METRICS_CSV}")


def main():
    args = parse_args()
    logger = setup_logging("compute_dino")
    conditions = CONDITIONS if args.all else [args.condition]
    for cond in conditions:
        sim = compute_dino_for(cond, logger)
        update_csv(cond, sim, logger)


if __name__ == "__main__":
    main()
