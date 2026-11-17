"""Fit on normal images, evaluate on the test split, save metrics + heatmap figures.

    python scripts/run.py --data data/synthetic --out outputs
    python scripts/run.py --data /path/to/mvtec_ad --categories bottle screw --out outputs_mvtec
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anomaly.data import list_categories, load_category  # noqa: E402
from anomaly.metrics import evaluate  # noqa: E402
from anomaly.model import PatchBankDetector  # noqa: E402


def save_figure(path: Path, data, maps, scores) -> None:
    """One row per defect type (first example) + one normal image: image | GT | heatmap | overlay."""
    lo, hi = np.percentile(maps, 1), np.percentile(maps, 99.9)
    names = ["good"] + [t for t in dict.fromkeys(data.defect_types) if t != "good"]
    rows = [data.defect_types.index(n) for n in names]
    fig, axes = plt.subplots(len(rows), 4, figsize=(10, 2.6 * len(rows)))
    for r, i in enumerate(rows):
        heat = np.clip((maps[i] - lo) / (hi - lo + 1e-9), 0, 1)
        axes[r, 0].imshow(data.test[i])
        axes[r, 1].imshow(data.masks[i], cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(heat, cmap="jet", vmin=0, vmax=1)
        axes[r, 3].imshow(data.test[i])
        axes[r, 3].imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[r, 0].set_ylabel(names[r], fontsize=11)
        axes[r, 3].set_xlabel(f"score {scores[i]:.2f}", fontsize=9)
    for c, t in enumerate(["image", "ground truth", "anomaly map", "overlay"]):
        axes[0, c].set_title(t)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--coreset", type=float, default=0.1, help="fraction of patches kept in memory bank")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for cat in a.categories or list_categories(a.data):
        data = load_category(a.data, cat, a.size)
        det = PatchBankDetector(a.stride, a.coreset, a.k, seed=a.seed)

        t0 = time.perf_counter()
        det.fit(data.train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        maps, scores = det.score(data.test)
        ms_per_img = (time.perf_counter() - t0) / len(data.test) * 1000

        res = evaluate(scores, maps, data.labels, data.masks, data.defect_types)
        res.update(n_train=len(data.train), n_test=len(data.test), bank_size=len(det.bank),
                   fit_seconds=round(fit_s, 1), ms_per_image=round(ms_per_img, 1))
        results[cat] = res
        save_figure(out / f"{cat}_heatmaps.png", data, maps, scores)
        print(f"{cat}: image AUROC {res['image_auroc']:.3f} | pixel AUROC {res['pixel_auroc']:.3f} | "
              f"pixel AP {res['pixel_ap']:.3f} | {ms_per_img:.0f} ms/img | fit {fit_s:.0f}s")

    mean = {m: float(np.mean([r[m] for r in results.values()])) for m in ("image_auroc", "pixel_auroc", "pixel_ap")}
    (out / "results.json").write_text(json.dumps({"config": vars(a), "mean": mean, "categories": results}, indent=2))
    print("mean:", {k: round(v, 3) for k, v in mean.items()})


if __name__ == "__main__":
    main()
