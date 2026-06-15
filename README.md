# Manga Style Diffusion

LoRA fine-tuning on Manga109-s to study how captioning strategy affects sub-genre style fidelity in noir manga generation.

**Research question:** Does captioning strategy determine sub-genre style fidelity when fine-tuning diffusion models on a niche manga aesthetic?

![Qualitative Comparison](paper/figures/qual_grid.png)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your API keys:
```
ANTHROPIC_API_KEY=your_key_here
WANDB_API_KEY=your_key_here
HF_TOKEN=your_key_here
```

## Project Structure

See `CLAUDE.md` for full structure and progress tracker.
