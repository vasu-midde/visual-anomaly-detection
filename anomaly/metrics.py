"""Image-level and pixel-level evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate(img_scores: np.ndarray, maps: np.ndarray, labels: np.ndarray,
             masks: np.ndarray, defect_types: list[str]) -> dict:
    types = np.array(defect_types)
    res = {
        "image_auroc": float(roc_auc_score(labels, img_scores)),
        "pixel_auroc": float(roc_auc_score(masks.ravel(), maps.ravel())),
        "pixel_ap": float(average_precision_score(masks.ravel(), maps.ravel())),
        "per_defect_image_auroc": {},
    }
    good = types == "good"
    for d in sorted(set(defect_types) - {"good"}):
        sel = good | (types == d)
        res["per_defect_image_auroc"][d] = float(roc_auc_score(labels[sel], img_scores[sel]))
    return res
