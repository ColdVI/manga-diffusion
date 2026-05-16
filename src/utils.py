"""Shared helpers for the manga-diffusion pipeline."""

import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from PIL import Image


def setup_logging(script_name: str, log_dir: str = "logs") -> logging.Logger:
    """Create a logger that writes to both stdout and a timestamped log file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger


def parse_manga109_page(page_elem) -> dict:
    """Extract frames and text boxes from a Manga109 XML page element.

    Returns dict with keys:
      index (int), width (int), height (int),
      frames (list of (xmin,ymin,xmax,ymax)),
      texts  (list of (xmin,ymin,xmax,ymax))
    """
    index = int(page_elem.get("index", 0))
    width = int(page_elem.get("width", 0))
    height = int(page_elem.get("height", 0))

    frames, texts = [], []
    for elem in page_elem:
        xmin = int(elem.get("xmin", 0))
        ymin = int(elem.get("ymin", 0))
        xmax = int(elem.get("xmax", 0))
        ymax = int(elem.get("ymax", 0))
        if elem.tag == "frame":
            frames.append((xmin, ymin, xmax, ymax))
        elif elem.tag == "text":
            texts.append((xmin, ymin, xmax, ymax))

    return {"index": index, "width": width, "height": height,
            "frames": frames, "texts": texts}


def compute_text_coverage(frame: tuple, texts: list) -> float:
    """Return fraction of frame area covered by text bounding boxes."""
    fx1, fy1, fx2, fy2 = frame
    frame_area = max(1, (fx2 - fx1) * (fy2 - fy1))
    covered = 0
    for tx1, ty1, tx2, ty2 in texts:
        ix1 = max(fx1, tx1)
        iy1 = max(fy1, ty1)
        ix2 = min(fx2, tx2)
        iy2 = min(fy2, ty2)
        if ix2 > ix1 and iy2 > iy1:
            covered += (ix2 - ix1) * (iy2 - iy1)
    return covered / frame_area


def resize_pad_to_square(img: Image.Image, size: int = 768,
                          bg_color: int = 255) -> Image.Image:
    """Resize image to fit inside size×size, pad remaining space with bg_color."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (size, size), (bg_color, bg_color, bg_color))
    offset_x = (size - new_w) // 2
    offset_y = (size - new_h) // 2
    canvas.paste(img, (offset_x, offset_y))
    return canvas
