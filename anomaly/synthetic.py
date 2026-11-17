"""Procedural texture images with synthetic defects, written in MVTec AD layout.

Lets the whole pipeline run offline. Two "categories" (woven fabric, brushed metal) and four
defect types (scratch, stain, crack, hole), each with a pixel-accurate ground-truth mask.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SIZE = 256
CATEGORIES = ("fabric", "metal")
DEFECTS = ("scratch", "stain", "crack", "hole")


def _texture(kind: str, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    if kind == "fabric":
        px, py = rng.uniform(0, 2 * np.pi, 2)
        weave = np.sin(2 * np.pi * x / 8 + px) * np.sin(2 * np.pi * y / 8 + py)
        base = 0.55 + 0.15 * weave + rng.normal(0, 0.02, (SIZE, SIZE))
        tint = np.array([0.95, 0.75, 0.55]) * rng.uniform(0.95, 1.05)
    else:
        streaks = cv2.GaussianBlur(rng.normal(size=(SIZE, SIZE)).astype(np.float32), (0, 0),
                                   sigmaX=25, sigmaY=0.8)
        base = 0.5 + 0.9 * streaks / streaks.std() * 0.08 + rng.normal(0, 0.01, (SIZE, SIZE))
        tint = np.array([0.75, 0.8, 0.9]) * rng.uniform(0.95, 1.05)
    return np.clip(base[..., None] * tint, 0, 1).astype(np.float32)


def _random_walk(rng, n: int, step: float) -> np.ndarray:
    pt = rng.uniform(50, SIZE - 50, 2)
    ang = rng.uniform(0, 2 * np.pi)
    pts = [pt.copy()]
    for _ in range(n):
        ang += rng.normal(0, 0.5)
        pt = np.clip(pt + step * np.array([np.cos(ang), np.sin(ang)]), 5, SIZE - 6)
        pts.append(pt.copy())
    return np.round(np.array(pts)).astype(np.int32)


def add_defect(img: np.ndarray, kind: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Returns (defective image float32 in [0,1], uint8 mask 0/255)."""
    mask = np.zeros((SIZE, SIZE), np.uint8)
    if kind == "scratch":
        p0 = rng.uniform(30, SIZE - 30, 2)
        ang = rng.uniform(0, np.pi)
        p1 = p0 + rng.uniform(50, 110) * np.array([np.cos(ang), np.sin(ang)])
        cv2.line(mask, tuple(map(int, p0)), tuple(map(int, np.clip(p1, 0, SIZE - 1))), 255, int(rng.integers(1, 3)))
        target, alpha = np.full(3, rng.choice([0.15, 0.95])), 0.75
    elif kind == "stain":
        c = tuple(int(v) for v in rng.uniform(50, SIZE - 50, 2))
        cv2.ellipse(mask, c, (int(rng.integers(10, 22)), int(rng.integers(8, 18))),
                    float(rng.uniform(0, 180)), 0, 360, 255, -1)
        target, alpha = rng.uniform(0.2, 0.9, 3), 0.55
    elif kind == "crack":
        cv2.polylines(mask, [_random_walk(rng, 14, 9)], False, 255, 2)
        target, alpha = np.full(3, 0.08), 0.85
    elif kind == "hole":
        c = tuple(int(v) for v in rng.uniform(40, SIZE - 40, 2))
        cv2.ellipse(mask, c, (int(rng.integers(5, 10)),) * 2, 0, 0, 360, 255, -1)
        target, alpha = np.full(3, 0.05), 0.95
    else:
        raise ValueError(kind)
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 0.8)[..., None]
    out = img * (1 - alpha * soft) + target.astype(np.float32) * alpha * soft
    return np.clip(out, 0, 1).astype(np.float32), mask


def _save(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor((img * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))


def generate(root: str | Path, n_train: int = 40, n_test_good: int = 20,
             n_per_defect: int = 10, seed: int = 0) -> None:
    root = Path(root)
    for ci, cat in enumerate(CATEGORIES):
        rng = np.random.default_rng(seed * 100 + ci)
        for i in range(n_train):
            _save(root / cat / "train" / "good" / f"{i:03d}.png", _texture(cat, rng))
        for i in range(n_test_good):
            _save(root / cat / "test" / "good" / f"{i:03d}.png", _texture(cat, rng))
        for d in DEFECTS:
            for i in range(n_per_defect):
                img, mask = add_defect(_texture(cat, rng), d, rng)
                _save(root / cat / "test" / d / f"{i:03d}.png", img)
                gt = root / cat / "ground_truth" / d / f"{i:03d}_mask.png"
                gt.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(gt), mask)
