"""
Generate evaluation samples from trained LoRA models.

For each condition (baseline, blip2, wd14):
  - Generates 100 images = 10 prompts x 10 seeds
  - Saves to evaluation/results/{condition}/generated/

Usage:
    python evaluation/generate_samples.py --condition blip2 --lora path/to/lora.safetensors
    python evaluation/generate_samples.py --condition wd14   --lora path/to/lora.safetensors
    python evaluation/generate_samples.py --condition baseline
    python evaluation/generate_samples.py --dry-run --condition blip2 --lora path/to/lora.safetensors
"""

import argparse
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import setup_logging

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

SEEDS = list(range(10))
RESULTS_DIR = Path("evaluation/results")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=["baseline", "blip2", "wd14"])
    p.add_argument("--lora", type=str, default=None,
                   help="Path to .safetensors LoRA file (not needed for baseline)")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate only 2 images to verify pipeline")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    return p.parse_args()


def load_pipeline(condition: str, lora_path: str | None, logger) -> StableDiffusionXLPipeline:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    logger.info(f"Loading SDXL on {device} ...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    if condition != "baseline":
        if not lora_path:
            raise ValueError(f"--lora required for condition '{condition}'")
        lora_path = Path(lora_path)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA not found: {lora_path}")
        logger.info(f"Loading LoRA from {lora_path} ...")
        pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
    return pipe, device


def generate_all(args, logger):
    pipe, device = load_pipeline(args.condition, args.lora, logger)
    out_dir = RESULTS_DIR / args.condition / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = EVAL_PROMPTS[:1] if args.dry_run else EVAL_PROMPTS
    seeds = SEEDS[:2] if args.dry_run else SEEDS
    total = len(prompts) * len(seeds)
    logger.info(f"Generating {total} images → {out_dir}")

    pbar = tqdm(total=total, desc=f"Generating ({args.condition})")
    for prompt_idx, prompt in enumerate(prompts):
        for seed in seeds:
            fname = out_dir / f"p{prompt_idx:02d}_s{seed:02d}.png"
            if fname.exists():
                pbar.update(1)
                continue
            generator = torch.Generator(device=device).manual_seed(seed)
            result = pipe(
                prompt=prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                height=768,
                width=768,
            )
            result.images[0].save(fname)
            pbar.update(1)

    pbar.close()
    count = len(list(out_dir.glob("*.png")))
    logger.info(f"Done — {count} images saved to {out_dir}")


def main():
    args = parse_args()
    logger = setup_logging("generate_samples")
    if args.condition != "baseline" and not args.lora:
        logger.error("--lora is required for non-baseline conditions")
        sys.exit(1)
    generate_all(args, logger)


if __name__ == "__main__":
    main()
