"""Step 2 - per-vertebra 2.5D ROIs from the GROUND-TRUTH masks + manifest.csv.
Uses GT masks on purpose: the classifier can be developed before the U-Net exists.
grades.csv must have columns: subject, vert_label, grade (0-3). Missing -> grade=-1 (unlabelled)."""
import argparse, pathlib, re, yaml, numpy as np, pandas as pd
from tqdm import tqdm
from vertfrac.preprocess import load_ct_and_mask, extract_vertebra_roi, VERT_NAMES, THORACOLUMBAR
from vertfrac.splits import patient_id, assign_folds, make_split_column, assert_no_leakage

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--limit", type=int, default=0, help="only first N scans (debugging)")
cfg = yaml.safe_load(open(ap.parse_args().config)); a = ap.parse_args()
d, p, sp = cfg["data"], cfg["preprocess"], cfg["splits"]
root, out = pathlib.Path(d["verse_root"]), pathlib.Path(d["roi_dir"]); out.mkdir(parents=True, exist_ok=True)

grades = {}
if pathlib.Path(d["grades_csv"]).exists():
    g = pd.read_csv(d["grades_csv"]); grades = {(r.subject, int(r.vert_label)): int(r.grade) for r in g.itertuples()}
else:
    print("WARNING: no grades.csv - all ROIs will be grade -1 (fine for a segmentation-only dry run)")

masks = {re.sub(r"_seg-vert_msk\.nii\.gz$", "", q.name): q for q in root.rglob("*_seg-vert_msk.nii.gz")}
rows = []
cts = sorted(root.rglob("*_ct.nii.gz"))[: a.limit or None]
for ct in tqdm(cts):
    sid = re.sub(r"_ct\.nii\.gz$", "", ct.name)
    if sid not in masks: continue
    ct_hu, mk = load_ct_and_mask(ct, masks[sid], p["spacing_mm"])
    for lab in np.unique(mk):
        if lab not in THORACOLUMBAR: continue          # chest-CT project: T/L only
        roi = extract_vertebra_roi(ct_hu, mk, int(lab), hu_min=p["hu_min"], hu_max=p["hu_max"],
                                   roi_size=p["roi_size"], offsets_mm=p["slice_offsets_mm"], spacing_mm=p["spacing_mm"])
        if roi is None: continue
        f = out / f"{sid}_{int(lab)}.npy"; np.save(f, roi.astype(np.float16))
        rows.append(dict(roi_path=str(f), subject=sid, patient=patient_id(sid), vert_label=int(lab),
                         vert_name=VERT_NAMES[int(lab)], grade=grades.get((patient_id(sid), int(lab)), -1)))
df = make_split_column(assign_folds(pd.DataFrame(rows), sp["n_folds"], sp["seed"]), sp["test_fold"], sp["val_fold"])
assert_no_leakage(df); df.to_csv(d["manifest"], index=False)
print(df.groupby(["split", "grade"]).size().unstack(fill_value=0))
