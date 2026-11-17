"""Loader for the MVTec AD folder layout (works for real MVTec AD and for the synthetic data).

    root/<category>/train/good/*.png
    root/<category>/test/good/*.png
    root/<category>/test/<defect_type>/*.png
    root/<category>/ground_truth/<defect_type>/<name>_mask.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class CategoryData:
    train: np.ndarray           # (N, S, S, 3) uint8 RGB, normal only
    test: np.ndarray            # (M, S, S, 3) uint8 RGB
    labels: np.ndarray          # (M,) 0 = normal, 1 = anomalous
    masks: np.ndarray           # (M, S, S) bool ground truth
    defect_types: list[str]     # (M,) defect name per test image ("good" for normal)


def _read_rgb(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA)


def _files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _EXTS)


def list_categories(root: str | Path) -> list[str]:
    return sorted(p.name for p in Path(root).iterdir() if (p / "train" / "good").is_dir())


def load_category(root: str | Path, category: str, size: int = 256,
                  max_train: int | None = None) -> CategoryData:
    base = Path(root) / category
    train_files = _files(base / "train" / "good")[:max_train]
    train = np.stack([_read_rgb(p, size) for p in train_files])

    imgs, labels, masks, types = [], [], [], []
    for sub in sorted((base / "test").iterdir()):
        if not sub.is_dir():
            continue
        for p in _files(sub):
            imgs.append(_read_rgb(p, size))
            if sub.name == "good":
                labels.append(0)
                masks.append(np.zeros((size, size), bool))
            else:
                labels.append(1)
                m = cv2.imread(str(base / "ground_truth" / sub.name / f"{p.stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
                masks.append(cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) > 0)
            types.append(sub.name)
    return CategoryData(train, np.stack(imgs), np.array(labels), np.stack(masks), types)
