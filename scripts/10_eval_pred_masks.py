"""Honest end-to-end evaluation: segment with YOUR U-Net (or use masks you already have), number the vertebrae,
match each predicted vertebra to the ground-truth vertebra (nearest centroid, must be < --max-dist mm), grade it with
the trained classifier, and compare with the GT Genant grade. Also reports how many GT vertebrae were missed.
   python scripts/10_eval_pred_masks.py --run runs/supcon --seg-ckpt runs/unet/best.pt --split test
Result: results/end2end_<split>.json  (compare with the GT-mask numbers from 04_train_classifier.py)"""
import argparse, json, pathlib, re, numpy as np, pandas as pd, torch
from vertfrac.infer import screen_volume, load_classifier, load_cfg
from vertfrac.preprocess import load_ct_and_mask
from vertfrac.metrics import grade_metrics

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--run", required=True); ap.add_argument("--seg-ckpt"); ap.add_argument("--pred-mask-dir", help="folder of <sid>_pred.nii.gz instead of a U-Net")
ap.add_argument("--split", default="test"); ap.add_argument("--max-dist", type=float, default=12.0); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args(); cfg = load_cfg(a.config); dev = "cuda" if torch.cuda.is_available() else "cpu"
clf = load_classifier(a.run, cfg, dev); man = pd.read_csv(cfg["data"]["manifest"]); man = man[(man.split == a.split) & (man.grade >= 0)]
root = pathlib.Path(cfg["data"]["verse_root"]); cts = {re.sub(r"_ct\.nii\.gz$", "", q.name): q for q in root.rglob("*_ct.nii.gz")}
gts = {re.sub(r"_seg-vert_msk\.nii\.gz$", "", q.name): q for q in root.rglob("*_seg-vert_msk.nii.gz")}
y_true, probs, missed, total, dists = [], [], 0, 0, []
sids = sorted(man.subject.unique())[: a.limit or None]
for sid in sids:
    kw = dict(mask_path=pathlib.Path(a.pred_mask_dir) / f"{sid}_pred.nii.gz") if a.pred_mask_dir else dict(seg_ckpt=a.seg_ckpt)
    res, ct_hu, inst = screen_volume(cts[sid], cfg, clf, device=dev, with_cam=False, **kw)
    _, gt_mask = load_ct_and_mask(cts[sid], gts[sid], cfg["preprocess"]["spacing_mm"])
    sp = cfg["preprocess"]["spacing_mm"]
    pred_c = np.array([r["centroid_xyz"] for r in res]) if res else np.zeros((0, 3))
    for _, g in man[man.subject == sid].iterrows():
        total += 1; pts = np.argwhere(gt_mask == g.vert_label)
        if len(pts) == 0 or len(pred_c) == 0: missed += 1; continue
        c = pts.mean(0); d = np.linalg.norm(pred_c - c, axis=1) * sp; j = int(d.argmin())
        if d[j] > a.max_dist: missed += 1; continue
        dists.append(float(d[j])); y_true.append(int(g.grade))
        probs.append([res[j]["p_normal"], res[j]["p_mild"], res[j]["p_moderate"], res[j]["p_severe"]])
rep = dict(split=a.split, scans=len(sids), gt_vertebrae=total, matched=len(y_true), missed=missed,
           mean_centroid_error_mm=float(np.mean(dists)) if dists else None,
           metrics=grade_metrics(y_true, np.array(probs)) if y_true else None)
pathlib.Path("results").mkdir(exist_ok=True); json.dump(rep, open(f"results/end2end_{a.split}.json", "w"), indent=2)
print(json.dumps({k: v for k, v in rep.items() if k != "metrics"}, indent=2)); print(rep["metrics"])
