"""Generate the synthetic dataset (MVTec AD layout):  python scripts/generate_synthetic.py --out data/synthetic"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anomaly.synthetic import generate  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-test-good", type=int, default=20)
    ap.add_argument("--n-per-defect", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(a.out, a.n_train, a.n_test_good, a.n_per_defect, a.seed)
    print(f"wrote dataset to {a.out}")
