"""Collect every runs/**/results.json into paper-ready tables and figures (mean +/- std over seeds).
Outputs (in results/): ablation.csv, ablation.tex, per_grade_f1.csv, confusion_<run>.png, roc.png.
Nothing is invented: every number comes from a results.json / test_probs.npz."""
import argparse, glob, json, pathlib, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="results")
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(exist_ok=True)
rows = []
for f in glob.glob(f"{a.runs}/**/results.json", recursive=True):
    r = json.load(open(f)); c, t = r["config"], r["test"]
    rows.append(dict(path=str(pathlib.Path(f).parent), lam=c["lam"], temp=c["temperature"], backbone=c["backbone"],
                     gamma=c["focal_gamma"], balanced=int(c["balanced"]), seed=c["seed"], macro_f1=t["macro_f1_4class"],
                     sens=t["sensitivity"], spec=t["specificity"], auc=t["auc_binary"], f1_bin=t["f1_binary"],
                     acc=t["accuracy_4class"], **{f"f1_g{i}": v for i, v in enumerate(t["per_class_f1"])},
                     n_test=str(t["n_per_grade"])))
assert rows, f"no results.json under {a.runs}"
df = pd.DataFrame(rows); keys = ["backbone", "lam", "temp", "gamma", "balanced"]
metrics = ["macro_f1", "sens", "spec", "auc", "f1_bin", "f1_g0", "f1_g1", "f1_g2", "f1_g3"]
g = df.groupby(keys); mean, std = g[metrics].mean(), g[metrics].std().fillna(0)
tab = mean.copy().astype(object)
for k in metrics: tab[k] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[k], std[k])]
tab["n_seeds"] = g.size(); tab["test_vertebrae_per_grade"] = g["n_test"].first()
tab = tab.reset_index(); tab.to_csv(out / "ablation.csv", index=False)
(out / "ablation.tex").write_text(tab[keys + ["macro_f1", "sens", "spec", "auc", "n_seeds"]].to_latex(index=False, escape=True))
tab[keys + ["f1_g0", "f1_g1", "f1_g2", "f1_g3"]].to_csv(out / "per_grade_f1.csv", index=False)
print(tab[keys + ["macro_f1", "sens", "spec", "auc", "n_seeds"]].to_string(index=False))

# confusion matrices + ROC for the best run of each lam (by test macro-F1)
fig, ax = plt.subplots(figsize=(5, 4))
for lam, d in df.groupby("lam"):
    best = d.sort_values("macro_f1").iloc[-1]; z = np.load(pathlib.Path(best.path) / "test_probs.npz")
    yb, s = (z["y"] >= 1).astype(int), 1 - z["p"][:, 0]
    if 0 < yb.sum() < len(yb):
        fpr, tpr, _ = roc_curve(yb, s); ax.plot(fpr, tpr, label=f"lambda={lam}")
    cm = np.array(json.load(open(pathlib.Path(best.path) / "results.json"))["test"]["confusion_4class"])
    f2, a2 = plt.subplots(figsize=(4, 4)); a2.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4): a2.text(j, i, cm[i, j], ha="center", va="center")
    a2.set_xticks(range(4)); a2.set_yticks(range(4)); a2.set_xlabel("predicted grade"); a2.set_ylabel("true grade")
    a2.set_title(f"lambda={lam} (best seed)"); f2.tight_layout(); f2.savefig(out / f"confusion_lam{lam}.png", dpi=150); plt.close(f2)
ax.plot([0, 1], [0, 1], "k--", lw=.5); ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity"); ax.legend()
ax.set_title("Binary: fractured vs normal"); fig.tight_layout(); fig.savefig(out / "roc.png", dpi=150)
print("wrote", out)
