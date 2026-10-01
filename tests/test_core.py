import numpy as np, pandas as pd, pytest
from vertfrac.preprocess import window_normalize, extract_vertebra_roi
from vertfrac.splits import assign_folds, make_split_column, assert_no_leakage, patient_id


def test_window_range():
    x = window_normalize(np.array([-1000., -200, 400, 1000, 3000]), -200, 1000)
    assert x.min() == 0 and x.max() == 1 and np.isclose(x[2], 0.5)


def test_roi_shape_and_missing_label():
    ct = np.random.uniform(-200, 1000, (40, 60, 80)).astype(np.float32)
    mk = np.zeros(ct.shape, np.int16); mk[15:25, 20:40, 30:50] = 12
    roi = extract_vertebra_roi(ct, mk, 12, hu_min=-200, hu_max=1000, roi_size=96)
    assert roi.shape == (3, 96, 96) and 0 <= roi.min() and roi.max() <= 1
    assert extract_vertebra_roi(ct, mk, 13, hu_min=-200, hu_max=1000) is None


def test_no_patient_leakage():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"subject": [f"sub-verse{i:03d}" for i in range(60) for _ in range(12)]})
    df["patient"] = df.subject.map(patient_id); df["grade"] = rng.choice(4, len(df), p=[.8, .1, .06, .04])
    df = make_split_column(assign_folds(df, 5, 0), 0, 1); assert_no_leakage(df)
    assert set(df.split) == {"train", "val", "test"}
