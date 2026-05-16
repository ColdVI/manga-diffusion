"""
Compute mean CLIP score (image-prompt cosine similarity) for generated images.

Usage:
    python evaluation/compute_clip.py --condition blip2
    python evaluation/compute_clip.py --all
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import setup_logging

RESULTS_DIR = Path("evaluation/results")
METRICS_CSV = RESULTS_DIR / "metrics_summary.csv"
CONDITIONS = ["baseline", "blip2", "wd14"]

EVAL_PROMPTS = [
    "gknoir style, heavy ink linework, close-up panel, detective in rain, tense atmosphere",
    "gknoir style, thin precise linework, wide establishing shot, empty city street, melancholic",
    "gknoir style, heavy ink linework, over-shoulder panel, figure in doorway, stark shadows",
    "gknoir style, bold ink, action panel, two figures confronting, high tension",
    "gknoir style, fine linework, interior scene, desk with lamp, quiet solitude",
    "gknoir style, heavy ink, extreme close-up, eyes in shadow, mysterious",
    "gknoir style, crosshatch shading, mid-shot, woman in coat, rainy street",
    "gknoir style, stark contrast, panel sequence feel, running figure, urgency",
    "gknoir style, screentone shading, thoughtful expression, window light, introspective",
    "gknoir style, deep blacks, architecture detail, alley at night, foreboding",
]


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--condition", choices=CONDITIONS)
    g.add_argument("--all", action="store_true")
    return p.parse_args()


def prompt_for_image(fname: str) -> str:
    """Extract prompt index from filename p{idx}_s{seed}.png."""
    m = re.match(r"p(\d+)_s\d+\.png", fname)
    if m:
        return EVAL_PROMPTS[int(m.group(1))]
    return EVAL_PROMPTS[0]


def compute_clip_for(condition: str, logger) -> float:
    gen_dir = RESULTS_DIR / condition / "generated"
    images = sorted(gen_dir.glob("*.png"))
    if not images:
        logger.error(f"No generated images at {gen_dir}")
        return float("nan")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[{condition}] Loading CLIP on {device} ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()

    scores = []
    for img_path in tqdm(images, desc=f"CLIP [{condition}]"):
        prompt = prompt_for_image(img_path.name)
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=[prompt], images=image, return_tensors="pt",
                           padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        score = torch.nn.functional.cosine_similarity(
            out.image_embeds, out.text_embeds
        ).item()
        scores.append(score)

    mean_score = sum(scores) / len(scores)
    logger.info(f"[{condition}] CLIP score = {mean_score:.4f}")
    return mean_score


def update_csv(condition: str, clip_score: float, logger):
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    if METRICS_CSV.exists():
        with open(METRICS_CSV) as f:
            for row in csv.DictReader(f):
                rows[row["run_name"]] = row

    if condition not in rows:
        rows[condition] = {"run_name": condition, "fid": "", "clip_score": "", "dino_sim": "", "notes": ""}
    rows[condition]["clip_score"] = f"{clip_score:.4f}"

    with open(METRICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "fid", "clip_score", "dino_sim", "notes"])
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)
    logger.info(f"CSV updated: {METRICS_CSV}")


def main():
    args = parse_args()
    logger = setup_logging("compute_clip")
    conditions = CONDITIONS if args.all else [args.condition]
    for cond in conditions:
        score = compute_clip_for(cond, logger)
        update_csv(cond, score, logger)


if __name__ == "__main__":
    main()
