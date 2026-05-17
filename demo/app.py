"""
Manga Diffusion — Gradio demo.

Tab 1: Text → manga panel (choose BLIP-2 LoRA, WD14 LoRA, or no LoRA)
Tab 2: Side-by-side comparison of all three conditions for the same prompt

Deploy to HuggingFace Spaces:
  1. Upload this file + requirements_demo.txt to a new Space (SDK: Gradio)
  2. Upload your .safetensors LoRA files and set BLIP2_LORA / WD14_LORA below
  3. Set hardware to at least T4 (GPU)
"""

import os
from pathlib import Path

import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline

# ── LoRA paths ────────────────────────────────────────────────────────────────
# Set these to the paths of your trained .safetensors files.
# On HuggingFace Spaces, upload the files and use relative paths.
BLIP2_LORA = os.environ.get("BLIP2_LORA", "loras/blip2-rank32.safetensors")
WD14_LORA  = os.environ.get("WD14_LORA",  "loras/wd14-rank32.safetensors")

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
TRIGGER    = "gknoir style"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# ── pipeline cache ────────────────────────────────────────────────────────────
_pipes: dict[str, StableDiffusionXLPipeline] = {}


def _get_pipe(condition: str) -> StableDiffusionXLPipeline:
    if condition in _pipes:
        return _pipes[condition]

    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=DTYPE, use_safetensors=True
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    if condition == "blip2" and Path(BLIP2_LORA).exists():
        pipe.load_lora_weights(str(Path(BLIP2_LORA).parent),
                               weight_name=Path(BLIP2_LORA).name)
    elif condition == "wd14" and Path(WD14_LORA).exists():
        pipe.load_lora_weights(str(Path(WD14_LORA).parent),
                               weight_name=Path(WD14_LORA).name)

    _pipes[condition] = pipe
    return pipe


def _build_prompt(user_prompt: str) -> str:
    p = user_prompt.strip()
    if not p.lower().startswith(TRIGGER):
        p = f"{TRIGGER}, {p}"
    return p


# ── generation function ───────────────────────────────────────────────────────
def generate(prompt: str, condition: str, guidance: float, steps: int, seed: int):
    full_prompt = _build_prompt(prompt)
    pipe = _get_pipe(condition)
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    result = pipe(
        prompt=full_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
        height=768,
        width=768,
    )
    return result.images[0], full_prompt


def compare(prompt: str, guidance: float, steps: int, seed: int):
    imgs = []
    labels = []
    for cond in ["baseline", "blip2", "wd14"]:
        img, fp = generate(prompt, cond, guidance, steps, seed)
        imgs.append(img)
        labels.append(f"{'No LoRA (baseline)' if cond == 'baseline' else cond.upper() + ' LoRA'}: {fp}")
    return imgs[0], labels[0], imgs[1], labels[1], imgs[2], labels[2]


# ── UI ────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    ["detective in rain, tense atmosphere, close-up panel"],
    ["empty city street at night, melancholic wide shot"],
    ["figure in doorway, stark shadows, over-shoulder angle"],
    ["two figures confronting, high tension, bold ink"],
    ["interior scene, desk with lamp, quiet solitude"],
]

with gr.Blocks(title="Manga Diffusion — gknoir LoRA") as demo:
    gr.Markdown(
        """# Manga Diffusion
**BLIP-2 vs WD14 captioning for noir manga LoRA fine-tuning**

Trigger word `gknoir style` is prepended automatically.
[GitHub](https://github.com/ColdVI/manga-diffusion)"""
    )

    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt_in = gr.Textbox(
                    label="Prompt (trigger word auto-prepended)",
                    placeholder="detective in rain, tense atmosphere, close-up panel",
                    lines=2,
                )
                condition = gr.Radio(
                    choices=["blip2", "wd14", "baseline"],
                    value="blip2",
                    label="LoRA condition",
                    info="blip2 = better style fidelity | wd14 = better prompt alignment | baseline = no LoRA",
                )
                with gr.Row():
                    guidance = gr.Slider(5, 15, value=7.5, step=0.5, label="Guidance scale")
                    steps    = gr.Slider(20, 50, value=30, step=5,   label="Steps")
                seed = gr.Number(value=42, label="Seed", precision=0)
                btn  = gr.Button("Generate", variant="primary")
                gr.Examples(examples=EXAMPLES, inputs=prompt_in)
            with gr.Column(scale=1):
                out_img    = gr.Image(label="Generated image", type="pil")
                out_prompt = gr.Textbox(label="Full prompt sent to model", interactive=False)

        btn.click(
            fn=generate,
            inputs=[prompt_in, condition, guidance, steps, seed],
            outputs=[out_img, out_prompt],
        )

    with gr.Tab("Compare all conditions"):
        gr.Markdown("Generates the same prompt with **no LoRA**, **BLIP-2 LoRA**, and **WD14 LoRA** side by side.")
        with gr.Row():
            cmp_prompt   = gr.Textbox(
                label="Prompt",
                value="detective in rain, tense atmosphere, close-up panel",
                lines=2,
            )
            cmp_guidance = gr.Slider(5, 15, value=7.5, step=0.5, label="Guidance scale")
            cmp_steps    = gr.Slider(20, 50, value=30, step=5,   label="Steps")
            cmp_seed     = gr.Number(value=42, label="Seed", precision=0)
        cmp_btn = gr.Button("Compare", variant="primary")
        with gr.Row():
            img_base  = gr.Image(label="Baseline (no LoRA)", type="pil")
            img_blip2 = gr.Image(label="BLIP-2 LoRA",        type="pil")
            img_wd14  = gr.Image(label="WD14 LoRA",          type="pil")
        with gr.Row():
            lbl_base  = gr.Textbox(label="", interactive=False)
            lbl_blip2 = gr.Textbox(label="", interactive=False)
            lbl_wd14  = gr.Textbox(label="", interactive=False)

        cmp_btn.click(
            fn=compare,
            inputs=[cmp_prompt, cmp_guidance, cmp_steps, cmp_seed],
            outputs=[img_base, lbl_base, img_blip2, lbl_blip2, img_wd14, lbl_wd14],
        )

if __name__ == "__main__":
    demo.launch()
