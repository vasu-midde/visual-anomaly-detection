"""Hand-crafted dense patch descriptors (no pretrained weights needed).

Each pixel gets 12 channels: 8 Gabor magnitudes (4 orientations x 2 scales),
3 Lab colour channels and 1 gradient magnitude. Channels are average-pooled to a
coarse grid and every grid cell is described together with its 3x3 neighbourhood,
which gives local context similar to PatchCore's neighbourhood aggregation.
"""

from __future__ import annotations

import cv2
import numpy as np

_THETAS = np.deg2rad([0, 45, 90, 135])
_WAVELENGTHS = (6.0, 12.0)


def _gabor_bank() -> list[np.ndarray]:
    bank = []
    for lam in _WAVELENGTHS:
        for theta in _THETAS:
            k = cv2.getGaborKernel((21, 21), sigma=0.56 * lam, theta=float(theta),
                                   lambd=lam, gamma=0.5, psi=0)
            bank.append((k / np.abs(k).sum()).astype(np.float32))
    return bank


_BANK = _gabor_bank()


def pixel_features(img_rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (H, W, 12) float32 feature map."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gabor = [np.abs(cv2.filter2D(gray, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)) for k in _BANK]
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return np.dstack(gabor + [lab, mag])


def patch_descriptors(img_rgb: np.ndarray, stride: int = 8) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (H/stride, W/stride, 12*9) float32 descriptors."""
    feats = pixel_features(img_rgb)
    h, w = img_rgb.shape[0] // stride, img_rgb.shape[1] // stride
    pooled = cv2.resize(feats, (w, h), interpolation=cv2.INTER_AREA)
    padded = np.pad(pooled, ((1, 1), (1, 1), (0, 0)), mode="edge")
    shifts = [padded[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)]
    return np.concatenate(shifts, axis=-1).astype(np.float32)
