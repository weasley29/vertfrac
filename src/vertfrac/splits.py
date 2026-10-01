"""Patient-level stratified splits (never split by vertebra: leakage)."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def patient_id(subject: str, pattern: str = r"^(sub-verse\d+)") -> str:
    """VerSe has 374 scans from 355 patients. Check in step 1.2 whether repeat scans share
    a subject id; if not, supply a patient-mapping and change `pattern`/this function."""
    m = re.match(pattern, subject)
    return m.group(1) if m else subject


def assign_folds(df: pd.DataFrame, n_folds=5, seed=42) -> pd.DataFrame:
    """Adds 'fold'. Stratifies on each vertebra's grade while keeping each patient in one fold."""
    df = df.copy()
    y = df["grade"].clip(lower=0).to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    col = df.columns.get_loc("fold")
    for k, (_, te) in enumerate(sgkf.split(df, y, groups=df["patient"])):
        df.iloc[te, col] = k
    return df


def make_split_column(df: pd.DataFrame, test_fold=0, val_fold=1) -> pd.DataFrame:
    df = df.copy()
    df["split"] = np.where(df.fold == test_fold, "test",
                   np.where(df.fold == val_fold, "val", "train"))
    return df


def assert_no_leakage(df: pd.DataFrame):
    g = df.groupby("patient")["split"].nunique()
    bad = g[g > 1]
    assert bad.empty, f"Patients appear in >1 split: {list(bad.index)[:5]}"
