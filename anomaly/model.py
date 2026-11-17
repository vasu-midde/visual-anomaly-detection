"""PatchCore-style detector: memory bank of normal patches + nearest-neighbour distance."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .features import patch_descriptors


def greedy_coreset(x: np.ndarray, m: int, seed: int = 0, proj_dim: int = 64) -> np.ndarray:
    """Greedy k-center subsampling (on a random projection) -> indices of m representative rows."""
    n = len(x)
    if m >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    z = x
    if x.shape[1] > proj_dim:
        z = x @ (rng.normal(size=(x.shape[1], proj_dim)) / np.sqrt(proj_dim)).astype(np.float32)
    selected = [int(rng.integers(n))]
    min_d = ((z - z[selected[0]]) ** 2).sum(1)
    for _ in range(m - 1):
        i = int(min_d.argmax())
        selected.append(i)
        min_d = np.minimum(min_d, ((z - z[i]) ** 2).sum(1))
    return np.array(selected)


class PatchBankDetector:
    """Fit on normal images only; score = distance of each patch to its nearest normal patch."""

    def __init__(self, stride: int = 8, coreset_ratio: float = 0.1, k: int = 1,
                 blur_sigma: float = 4.0, seed: int = 0):
        self.stride, self.coreset_ratio, self.k = stride, coreset_ratio, k
        self.blur_sigma, self.seed = blur_sigma, seed

    def _flat(self, img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        d = patch_descriptors(img, self.stride)
        return d.reshape(-1, d.shape[-1]), d.shape[:2]

    def fit(self, train_imgs: np.ndarray) -> "PatchBankDetector":
        flat = np.concatenate([self._flat(im)[0] for im in train_imgs])
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-6
        flat = (flat - self.mu) / self.sd
        idx = greedy_coreset(flat, max(1, int(len(flat) * self.coreset_ratio)), self.seed)
        self.bank = flat[idx]
        self._nn = NearestNeighbors(n_neighbors=self.k).fit(self.bank)
        return self

    def anomaly_map(self, img: np.ndarray) -> np.ndarray:
        flat, (h, w) = self._flat(img)
        dist, _ = self._nn.kneighbors((flat - self.mu) / self.sd)
        patch_scores = dist.mean(1).reshape(h, w).astype(np.float32)
        full = cv2.resize(patch_scores, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(full, (0, 0), self.blur_sigma)

    def score(self, imgs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (maps (N,H,W), image_scores (N,)); image score = max of the anomaly map."""
        maps = np.stack([self.anomaly_map(im) for im in imgs])
        return maps, maps.reshape(len(maps), -1).max(1)
