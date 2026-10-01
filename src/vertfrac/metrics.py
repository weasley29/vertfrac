import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score


def grade_metrics(y_true, prob):
    """y_true: (N,) grades 0-3; prob: (N,4) softmax. Reports 4-class macro-F1 and the binary
    (fractured = grade>=1) task, plus per-grade counts (always report them: heavy imbalance)."""
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    pred = prob.argmax(1)
    yb, pb = (y_true >= 1).astype(int), (pred >= 1).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, pb, labels=[0, 1]).ravel()
    out = {
        "n_per_grade": np.bincount(y_true, minlength=4).tolist(),
        "macro_f1_4class": float(f1_score(y_true, pred, average="macro", labels=[0, 1, 2, 3], zero_division=0)),
        "per_class_f1": f1_score(y_true, pred, average=None, labels=[0, 1, 2, 3], zero_division=0).tolist(),
        "confusion_4class": confusion_matrix(y_true, pred, labels=[0, 1, 2, 3]).tolist(),
        "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(precision_score(yb, pb, zero_division=0)),
        "recall": float(recall_score(yb, pb, zero_division=0)),
        "f1_binary": float(f1_score(yb, pb, zero_division=0)),
        "accuracy_4class": float((pred == y_true).mean()),
    }
    out["auc_binary"] = float(roc_auc_score(yb, 1.0 - prob[:, 0])) if 0 < yb.sum() < len(yb) else float("nan")
    return out
