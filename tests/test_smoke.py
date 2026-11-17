import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anomaly.data import list_categories, load_category  # noqa: E402
from anomaly.metrics import evaluate  # noqa: E402
from anomaly.model import PatchBankDetector  # noqa: E402
from anomaly.synthetic import generate  # noqa: E402


def test_pipeline_end_to_end(tmp_path):
    generate(tmp_path, n_train=10, n_test_good=6, n_per_defect=3, seed=1)
    assert list_categories(tmp_path) == ["fabric", "metal"]

    data = load_category(tmp_path, "fabric")
    assert data.train.shape == (10, 256, 256, 3)
    assert data.test.shape[0] == 6 + 4 * 3
    assert data.masks[data.labels == 1].any(axis=(1, 2)).all()   # every defect has a mask
    assert not data.masks[data.labels == 0].any()                # good images have none

    det = PatchBankDetector(coreset_ratio=0.1).fit(data.train)
    maps, scores = det.score(data.test)
    assert maps.shape == data.masks.shape

    res = evaluate(scores, maps, data.labels, data.masks, data.defect_types)
    assert res["image_auroc"] > 0.9
    assert res["pixel_auroc"] > 0.9
