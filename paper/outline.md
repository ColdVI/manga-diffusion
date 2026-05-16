# Paper Outline — Captioning Strategy as a Determinant of Style Fidelity in LoRA-based Manga Sub-genre Generation

**Target venue:** CVPR 2026 Workshop on Generative Models for Creative Content (4-page format)

## Status
- [x] Outline complete
- [x] main.tex drafted (all sections except Results and Discussion filled with placeholders)
- [x] references.bib complete (12 entries)
- [ ] Quantitative results table filled (waiting on evaluation runs)
- [ ] Qualitative figure generated (waiting on evaluation runs)
- [ ] Discussion and Conclusion written
- [ ] Abstract revised to include actual findings

## Section Map (main.tex)

### Abstract (drafted, needs final findings)
Problem → Method → Finding → Implication

### 1. Introduction (complete)
- Diffusion models fail on manga sub-genres
- LoRA as practical fine-tuning approach
- Caption strategy as unexamined variable
- 3 contributions: curation protocol, captioning comparison, evaluation benchmark

### 2. Related Work (complete)
- 2.1 Fine-Tuning: DreamBooth vs Textual Inversion vs LoRA trade-offs
- 2.2 Captioning: BLIP-2 (free-form, photorealistic bias) vs WD14 (tag-based, Danbooru domain)
- 2.3 Manga generation: Manga109, DiffSensei — gap statement that no prior work isolates caption strategy

### 3. Method (complete)
- 3.1 Dataset curation: XML-based panel extraction, noir selection criteria, 200 train / 40 test
- 3.2 Captioning conditions: trigger "gknoir style", BLIP-2 pipeline, WD14 ONNX pipeline
- 3.3 LoRA training: SDXL, rank=32, alpha=16, AdamW8bit, 2000 steps, A100 40GB
- 3.4 Evaluation: FID (lower better), CLIP score (higher better), DINOv2 style sim (higher better)

### 4. Experiments (placeholder tables/figures)
- Table 1: FID / CLIP / DINO for baseline, BLIP-2, WD14
- Figure 1: Qualitative comparison grid (3 prompts × 3 conditions)
- Table 2: Fixed evaluation prompts

### 5. Discussion and Conclusion (placeholder)
- Which condition wins and why (cross-attention token granularity argument)
- Limitations: dataset size, single sub-genre, rank ablation deferred
- Release: curation code + caption files + evaluation scripts

## Key References (all in references.bib)
- Ho et al. 2020 — DDPM
- Rombach et al. 2022 — Latent Diffusion / Stable Diffusion
- Podell et al. 2023 — SDXL (arXiv:2307.01952)
- Hu et al. 2021 — LoRA (arXiv:2106.09685)
- Ruiz et al. 2023 — DreamBooth (arXiv:2208.12242)
- Gal et al. 2022 — Textual Inversion (arXiv:2208.01618)
- Li et al. 2023 — BLIP-2 (arXiv:2301.12597)
- Radford et al. 2021 — CLIP (arXiv:2103.00020)
- Oquab et al. 2023 — DINOv2 (arXiv:2304.07193)
- Heusel et al. 2017 — FID
- Matsui et al. 2017 — Manga109
- Chen et al. 2024 — DiffSensei (arXiv:2412.07589)

## Next Steps (after WD14 training + evaluation complete)
1. Run `evaluation/generate_samples.py` for both conditions + baseline
2. Run `evaluation/compute_fid.py`, `compute_clip.py`, `compute_dino.py`
3. Fill Table 1 in main.tex with real numbers
4. Generate qualitative grid with `paper/figures/qual_grid.png`
5. Write Discussion paragraph based on findings
6. Revise abstract last sentence with actual result
