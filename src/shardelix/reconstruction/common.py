"""Shared types, coordinate convention and image I/O.

The pose is (x, y, k): top-left of the cropped raster after k clockwise
quarter-turns. Alpha==255 is a valid pixel; invalid RGB is always zero.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps


@dataclass(slots=True)
class Fragment:
    id: str
    rgba: np.ndarray
    input_origin: tuple[int,int] = (0,0)  # crop offset within original file

    @property
    def mask(self) -> np.ndarray:
        return self.rgba[:, :, 3] > 0

    def rotated(self, k: int) -> np.ndarray:
        return np.rot90(self.rgba, -k % 4)


@dataclass(frozen=True, slots=True)
class Pose:
    x: int
    y: int
    k: int = 0

    def as_dict(self) -> dict:
        return {"x": int(self.x), "y": int(self.y), "k": int(self.k % 4)}


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        json.dump(value,f,indent=2,allow_nan=False)


def read_source(path: Path, max_side: int = 0, page: int = 1) -> np.ndarray:
    """Images or a PDF page; transparent source pixels are flattened on white."""
    if path.suffix.lower() == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PDF support requires the optional 'pdf' dependency.") from exc
        with fitz.open(path) as doc:
            if not 1 <= page <= len(doc):
                raise ValueError("PDF page is out of range (1-based)")
            p = doc[page - 1].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", (p.width, p.height), p.samples)
    else:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGBA")
            bg = Image.new("RGBA", image.size, "white")
            bg.alpha_composite(image)
            image = bg.convert("RGB")
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.array(image)


def load_fragments(folder: Path, *, max_fragments=None, max_input_pixels=None) -> list[Fragment]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".webp", ".bmp"})
    from shardelix.reconstruction.limits import GRID_MAX, MAX_PIXELS
    limit = GRID_MAX if max_fragments is None else min(GRID_MAX, max_fragments)
    pixel_limit = MAX_PIXELS*3 if max_input_pixels is None else min(MAX_PIXELS*3, max_input_pixels)
    if not 1 <= len(files) <= limit:
        raise ValueError(f"Expected 1–{limit} individual image fragments")
    out = []
    total=0
    for p in files:
        with Image.open(p) as im:
            total+=im.width*im.height
            if total>pixel_limit:
                raise ValueError("Combined fragment crop pixels exceed resource budget")
            a = np.asarray(ImageOps.exif_transpose(im).convert("RGBA")).copy()
        m = a[:, :, 3] > 0
        if not m.any():
            raise ValueError(f"Empty fragment: {p}")
        yy, xx = np.nonzero(m)
        a = a[yy.min():yy.max()+1, xx.min():xx.max()+1].copy()
        a[~(a[:, :, 3] > 0)] = 0
        out.append(Fragment(p.stem, a,(int(xx.min()),int(yy.min()))))
    if len({p.id for p in out}) != len(out):
        raise ValueError("Fragment filenames must have unique stems")
    return out


def render(pieces: Iterable[Fragment], poses: dict[str, Pose], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, int]:
    """Return RGBA render, raster label map, and count of overwritten pixels."""
    h, w = shape
    canvas = np.zeros((h, w, 4), np.uint8)
    labels = np.full((h, w), -1, dtype=np.int32)
    collision = 0
    for index, p in enumerate(pieces):
        if p.id not in poses:
            continue
        pose = poses[p.id]
        a = p.rotated(pose.k)
        ph, pw = a.shape[:2]
        x, y = pose.x, pose.y
        if x < 0 or y < 0 or x + pw > w or y + ph > h:
            continue
        m = a[:, :, 3] > 0
        label = labels[y:y+ph, x:x+pw]
        collision += int(np.count_nonzero((label >= 0) & m))
        label[m] = index
        canvas[y:y+ph, x:x+pw][m] = a[m]
    return canvas, labels, collision


def save_rgba(path: Path, rgba: np.ndarray, *, compress_level: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(path, compress_level=compress_level)


def normalize_pose_set(pieces: list[Fragment], poses: dict[str, Pose]) -> tuple[dict[str, Pose], tuple[int, int]]:
    if not poses:
        return poses, (1, 1)
    mapping = {p.id: p for p in pieces}
    minx = min(p.x for p in poses.values())
    miny = min(p.y for p in poses.values())
    normalized = {i: Pose(p.x-minx, p.y-miny, p.k) for i, p in poses.items()}
    w = max(p.x + mapping[i].rotated(p.k).shape[1] for i, p in normalized.items())
    h = max(p.y + mapping[i].rotated(p.k).shape[0] for i, p in normalized.items())
    return normalized, (h, w)
