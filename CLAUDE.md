# Manga Diffusion Project — Master Instruction Set
> This file is the single source of truth for Claude Code / Copilot.
> Read this before touching any file. Update the status checkboxes as work completes.
> Never delete sections — mark them done instead.

---

## 0. Project Identity

**Project name:** Manga Style Diffusion  
**One-line summary:** LoRA fine-tuning on Manga109-s to study how captioning strategy affects sub-genre style fidelity in noir manga generation.  
**Research question (primary):** Does captioning strategy determine sub-genre style fidelity when fine-tuning diffusion models on a niche manga aesthetic?  
**Owner:** Anıl  
**Captioning tools:** BLIP-2 (auto) + WD14 tagger (auto) — no external API needed  
**Base diffusion model:** `stabilityai/stable-diffusion-xl-base-1.0`  
**LoRA library:** kohya_ss (training) + HuggingFace PEFT (inference)  
**Trigger word:** `gknoir` (short, unique, base model does not know this token)

---

## 1. Repository Structure

```
manga-diffusion-project/
├── CLAUDE.md                  ← this file (always read first)
├── README.md                  ← public-facing project description
├── .gitignore
├── requirements.txt           ← local dev dependencies
├── requirements_colab.txt     ← Colab/Kaggle training dependencies
│
├── data/
│   ├── raw/                   ← Manga109-s original files (gitignored)
│   ├── processed/             ← resized, cleaned panels (gitignored)
│   ├── captions/
│   │   ├── blip2/             ← auto-generated BLIP-2 captions (.txt per image)
│   │   └── wd14/              ← WD14 tagger output (.txt per image)
│   ├── train_set/             ← symlinks or copies for training (gitignored)
│   └── test_set/              ← held-out images, NEVER used in training (gitignored)
│
├── src/
│   ├── preprocess.py          ← image cleaning, resizing, panel extraction
│   ├── caption_blip2.py       ← BLIP-2 auto captioning
│   ├── caption_wd14.py        ← WD14 tagger captioning
│   ├── build_dataset.py       ← assemble final train/test split
│   └── utils.py               ← shared helpers
│
├── training/
│   ├── configs/
│   │   ├── lora_rank16.toml   ← kohya config, rank 16
│   │   ├── lora_rank32.toml   ← kohya config, rank 32
│   │   └── lora_rank64.toml   ← kohya config, rank 64
│   ├── train_colab.ipynb      ← Colab notebook that runs kohya training
│   └── outputs/               ← saved LoRA weights (gitignored)
│
├── evaluation/
│   ├── generate_samples.py    ← batch image generation from trained LoRA
│   ├── compute_fid.py         ← FID score against test set
│   ├── compute_clip.py        ← CLIP score for prompt alignment
│   ├── compute_dino.py        ← DINO feature similarity for style
│   └── results/               ← evaluation outputs, CSV tables, sample grids
│
├── demo/
│   ├── app.py                 ← Gradio app (sketch → manga panel)
│   └── requirements_demo.txt
│
└── paper/
    ├── outline.md             ← paper structure and notes
    ├── references.bib         ← BibTeX references
    └── figures/               ← diagrams, result grids for paper
```

---

## 2. Environment Setup

### Local (macOS — code editing, captioning, evaluation)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**requirements.txt contents:**
```
diffusers>=0.27.0
transformers>=4.40.0
accelerate>=0.30.0
torch>=2.2.0
pillow>=10.0.0
tqdm>=4.66.0
wandb>=0.17.0
datasets>=2.19.0
open-clip-torch>=2.24.0
pytorch-fid>=0.3.0
gradio>=4.0.0
```

### Colab/Kaggle (GPU training only)
```bash
pip install -r requirements_colab.txt
```

**requirements_colab.txt contents:**
```
diffusers>=0.27.0
transformers>=4.40.0
accelerate>=0.30.0
peft>=0.10.0
bitsandbytes>=0.43.0
wandb>=0.17.0
xformers
```

### Environment variables (create .env file, never commit)
```
WANDB_API_KEY=your_key_here
HF_TOKEN=your_key_here
```

---

## 3. Data Pipeline

### 3.1 What we have
- **Manga109-s**: approved access, files in `data/raw/`
- **Manga109** (full): application submitted, access pending — do not block on this

### 3.2 Preprocessing rules (implement in `src/preprocess.py`)
- Input: raw manga page scans (variable resolution, mixed formats)
- Output: individual panels in `data/processed/`, 768×768 px, RGB, PNG
- Rules:
  - Crop individual panels from full pages (use contour detection or manual crop grid)
  - Skip pages with heavy text overlay (speech bubbles covering >40% of panel)
  - Convert grayscale to RGB (manga is grayscale — SD expects RGB, replicate channels)
  - Normalize filename: `{work_id}_{page:03d}_{panel:02d}.png`
  - Log skipped images and reason to `data/processed/skip_log.txt`
- Target: 150–200 clean panels for training, 30–40 held out for test

### 3.3 Train/test split (implement in `src/build_dataset.py`)
- **Split by work, not by page** — all pages from one manga go entirely to train or test
- This prevents data leakage (same artist style appearing in both splits)
- 80% train works / 20% test works
- Fix random seed = 42 for reproducibility
- Write split manifest to `data/split_manifest.json`
- **Never shuffle after this point — the test set is frozen**

### 3.4 Caption strategies (two conditions — this is the experiment)

**Condition A — BLIP-2 auto captions** (`src/caption_blip2.py`)
- Load `Salesforce/blip2-opt-2.7b`
- Generate one caption per image, no prompt engineering
- Save as `data/captions/blip2/{image_stem}.txt`
- Example output: `"a black and white drawing of a man in a hat"`

**Condition B — WD14 tagger** (`src/caption_wd14.py`)
- Use `SmilingWolf/wd-v1-4-convnext-tagger-v2`
- Output Danbooru-style tags, threshold 0.35
- Save as `data/captions/wd14/{image_stem}.txt`
- Example output: `"1boy, monochrome, hat, noir, dramatic lighting, screentone"`

---

## 4. Training

### 4.1 Experiment matrix
We train 2 LoRA models, one per captioning condition. All other hyperparameters identical.

| Run | Caption source | Rank | Steps | LR | W&B run name |
|-----|---------------|------|-------|----|--------------|
| A | BLIP-2 | 32 | 2000 | 1e-4 | `blip2-rank32` |
| B | WD14 | 32 | 2000 | 1e-4 | `wd14-rank32` |

After primary results, ablation runs if time permits (using winning condition):
| Run | Caption source | Rank | Steps | Notes |
|-----|---------------|------|-------|-------|
| W-r16 | winning condition | 16 | 2000 | rank ablation |
| W-r64 | winning condition | 64 | 2000 | rank ablation |
| W-s1000 | winning condition | 32 | 1000 | steps ablation |
| W-s3000 | winning condition | 32 | 3000 | steps ablation |

### 4.2 kohya config template (write to `training/configs/lora_rank32.toml`)
```toml
[general]
enable_bucket = true
shuffle_caption = true
caption_extension = ".txt"
keep_tokens = 1

[dataset]
resolution = 768
batch_size = 1

[network]
network_module = "networks.lora"
network_dim = 32
network_alpha = 16

[optimizer]
optimizer_type = "AdamW8bit"
learning_rate = 1e-4
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 100

[training]
max_train_steps = 2000
save_every_n_steps = 500
mixed_precision = "fp16"
xformers = true
seed = 42

[logging]
log_with = "wandb"
wandb_run_name = "claude-rank32"

[model]
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
vae = "madebyollin/sdxl-vae-fp16-fix"
```

### 4.3 Colab notebook structure (`training/train_colab.ipynb`)
Cells in order:
1. Install dependencies
2. Mount Google Drive (save outputs there)
3. Clone this repo / upload training data
4. Set environment variables (WANDB key)
5. Run kohya training command
6. Save LoRA weights to Drive
7. Log completion to W&B

---

## 5. Evaluation

### 5.1 Generate samples (`evaluation/generate_samples.py`)
- For each trained LoRA (A, B):
  - Generate 100 images using 10 fixed prompts × 10 seeds
  - Fixed prompts are standardized — same for all conditions:
    ```python
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
    ```
  - Save to `evaluation/results/{run_name}/generated/`
- Also generate 100 images with **base SDXL (no LoRA)** as baseline

### 5.2 Metrics to compute

**FID** (`evaluation/compute_fid.py`)
- Compare generated distribution vs. test set distribution
- Use `pytorch-fid` library
- Run: `python -m pytorch_fid path/to/generated path/to/test_set`
- Lower = better

**CLIP score** (`evaluation/compute_clip.py`)
- For each generated image, compute cosine similarity between image embedding and its prompt
- Use `openai/clip-vit-large-patch14`
- Average across all 100 images per condition
- Higher = better

**DINO style similarity** (`evaluation/compute_dino.py`)
- Extract DINO-v2 features from generated images and test set images
- Compute mean cosine similarity between distributions
- More sensitive to style than FID
- Higher = better

**Save all results to** `evaluation/results/metrics_summary.csv`:
```
run_name, fid, clip_score, dino_sim, notes
baseline, X, X, X, base SDXL no LoRA
blip2-rank32, X, X, X, condition A
wd14-rank32, X, X, X, condition B
```

---

## 6. Demo App (`demo/app.py`)

Gradio interface with two tabs:

**Tab 1 — Text to manga panel**
- Text prompt input
- Trigger word auto-prepended
- Guidance scale slider (5–15)
- Steps slider (20–50)
- Seed input
- Output image

**Tab 2 — Sketch to manga panel**
- Sketchpad canvas input
- ControlNet lineart preprocessing
- Same controls as Tab 1
- Output image

Deploy to HuggingFace Spaces. Space URL goes in paper and README.

---

## 7. Paper Outline (`paper/outline.md`)

**Title (working):** "BLIP-2 vs Tag-based Captioning: Effect on Style Fidelity in LoRA Fine-tuning for Noir Manga Generation"

**Target venue:** CVPR 2026 Workshop on Generative Models for Creative Content (4-page format)

### Section structure

**Abstract** (150 words)  
Problem → Method → Finding → Implication

**1. Introduction** (0.5 page)
- Diffusion models generate generic "anime/manga" but fail on sub-genres
- Captioning is understudied as a variable in style-specific fine-tuning
- We compare two strategies on noir manga (Manga109-s)
- Contributions: curation protocol, captioning comparison, evaluation benchmark

**2. Related Work** (0.5 page)
- Diffusion fine-tuning: LoRA [Hu 2021], DreamBooth [Ruiz 2022], Textual Inversion [Gal 2022]
- Style in generative models: Gatys 2016, CLIP [Radford 2021]
- Manga datasets and generation: Manga109 [Matsui 2017], DiffSensei [2024]

**3. Method** (1 page)
- 3.1 Dataset curation from Manga109-s (noir sub-genre selection criteria)
- 3.2 Two captioning strategies (A: BLIP-2, B: WD14)
- 3.3 LoRA training setup (same hyperparams across conditions)
- 3.4 Evaluation protocol

**4. Experiments** (1.5 pages)
- 4.1 Quantitative results table (FID / CLIP / DINO per condition)
- 4.2 Qualitative comparison grids (same prompt, 3 conditions side by side)
- 4.3 Ablation: rank and training steps effect on best condition

**5. Discussion & Conclusion** (0.5 page)
- Which captioning strategy wins and why (hypothesis)
- Limitations: dataset size, single sub-genre, compute constraints
- Future work: SDXL vs SD1.5, other sub-genres, larger datasets

### Key references to add to `references.bib`
```
Ho et al. 2020 — DDPM
Rombach et al. 2022 — Latent Diffusion / Stable Diffusion
Ho & Salimans 2022 — Classifier-free guidance
Hu et al. 2021 — LoRA
Ruiz et al. 2022 — DreamBooth
Gal et al. 2022 — Textual Inversion
Radford et al. 2021 — CLIP
Matsui et al. 2017 — Manga109
Cohn 2013 — Visual Language of Comics (optional)
DiffSensei 2024 — manga generation (arxiv:2412.19303)
```

---

## 8. Progress Tracker

Claude Code: update this section as tasks complete. Change `[ ]` to `[x]`.

### Week 1 — Data & Captioning
- [x] Project folder and git initialized
- [x] Python venv created, requirements installed
- [x] `.env` file created with API keys
- [ ] W&B project created and login confirmed
- [x] Manga109-s files moved to `data/raw/`
- [x] `src/preprocess.py` written and tested
- [x] Train/test split generated, `data/split_manifest.json` saved
- [x] `src/caption_blip2.py` written and run on train set
- [x] `src/caption_wd14.py` written and run on train set
- [x] Caption quality spot-checked manually (10 samples per condition)
- [x] `data/train_set/` assembled with images + caption .txt files

### Week 2 — Training
- [ ] kohya_ss installed on Colab
- [x] `training/configs/lora_rank32.toml` created for all 2 conditions
- [x] `training/train_colab.ipynb` created and tested
- [x] Run A (BLIP-2) trained, weights saved
- [x] Run B (WD14) trained, weights saved
- [ ] All 2 runs visible in W&B with loss curves
- [ ] Qualitative spot-check: generate 5 images per LoRA, compare visually

### Week 3 — Evaluation & Demo
- [x] `evaluation/generate_samples.py` written and run for all conditions (use eval_colab.ipynb)
- [x] FID computed for all conditions
- [x] CLIP score computed for all conditions
- [x] DINO similarity computed for all conditions
- [x] `evaluation/results/metrics_summary.csv` populated
- [ ] Qualitative comparison grid images saved to `paper/figures/`
- [ ] Gradio demo app written (`demo/app.py`)
- [ ] Demo deployed to HuggingFace Spaces
- [ ] README updated with Space URL and result preview

### Week 4 — Paper & Polish (buffer)
- [ ] Ablation runs completed (rank / steps)
- [x] `paper/outline.md` written (structure + reference list)
- [x] `paper/main.tex` drafted (Intro, Related Work, Method complete; Results/Discussion are placeholders pending evaluation)
- [x] `paper/references.bib` complete (12 entries)
- [x] Quantitative results table filled in main.tex
- [x] Qualitative figure `paper/figures/qual_grid.png` generated
- [x] Abstract final sentence written (BLIP-2 wins FID+DINO, WD14 wins CLIP)
- [ ] All figures finalized
- [ ] Paper submitted to target workshop (or arXiv preprint)

---

## 9. Conventions Claude Code Must Follow

- **Always activate venv** before running any python script: `source venv/bin/activate`
- **Every script must have a `--dry-run` flag** that processes 3 images only (for testing)
- **Log everything** — every script writes a log file to `logs/{script_name}_{timestamp}.log`
- **W&B logging** — any training or evaluation script logs metrics to W&B
- **Reproducibility** — every random operation uses `seed=42` unless explicitly noted
- **Comments** — every function has a docstring explaining what it does, inputs, outputs
- **Error handling** — wrap API calls in try/except, skip failed images, log the failure
- **Progress bars** — use `tqdm` for any loop over images

---

## 10. How to Use This File With Claude Code

When starting a new session in Claude Code, say:

> "Read CLAUDE.md and tell me what the next unchecked task is in the Progress Tracker."

Claude Code will read this file, find the next `[ ]` item, and implement it.

After completing a task, say:

> "Mark that task done in CLAUDE.md and show me what's next."

This keeps the entire project state in one readable file that any AI assistant can pick up.

---

*Last updated: Week 1 complete — all data, captions, and training configs ready*
