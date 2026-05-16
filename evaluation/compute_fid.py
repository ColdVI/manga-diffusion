"""
Compute FID between generated images and the test set.

Usage:
    python evaluation/compute_fid.py --condition blip2
    python evaluation/compute_fid.py --condition wd14
    python evaluation/compute_fid.py --condition baseline
    python evaluation/compute_fid.py --all   # runs all three and writes CSV
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from pytorch_fid import fid_score

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
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--dims", type=int, default=2048)
    return p.parse_args()


def compute_fid_for(condition: str, batch_size: int, dims: int, logger) -> float:
    gen_dir = RESULTS_DIR / condition / "generated"
    if not gen_dir.exists() or not list(gen_dir.glob("*.png")):
        logger.error(f"No generated images for {condition} at {gen_dir}")
        return float("nan")
    if not TEST_SET_DIR.exists() or not list(TEST_SET_DIR.glob("*.png")):
        logger.error(f"Test set not found at {TEST_SET_DIR}")
        return float("nan")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[{condition}] Computing FID on {device} ...")
    fid = fid_score.calculate_fid_given_paths(
        [str(gen_dir), str(TEST_SET_DIR)],
        batch_size=batch_size,
        device=device,
        dims=dims,
    )
    logger.info(f"[{condition}] FID = {fid:.4f}")
    return fid


def update_csv(condition: str, fid: float, logger):
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    if METRICS_CSV.exists():
        with open(METRICS_CSV) as f:
            for row in csv.DictReader(f):
                rows[row["run_name"]] = row

    if condition not in rows:
        rows[condition] = {"run_name": condition, "fid": "", "clip_score": "", "dino_sim": "", "notes": ""}
    rows[condition]["fid"] = f"{fid:.4f}"

    with open(METRICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "fid", "clip_score", "dino_sim", "notes"])
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)
    logger.info(f"CSV updated: {METRICS_CSV}")


def main():
    args = parse_args()
    logger = setup_logging("compute_fid")
    conditions = CONDITIONS if args.all else [args.condition]
    for cond in conditions:
        fid = compute_fid_for(cond, args.batch_size, args.dims, logger)
        update_csv(cond, fid, logger)


if __name__ == "__main__":
    main()
