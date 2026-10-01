"""Screen ONE CT volume and save a CSV + overlay figure.
   python scripts/11_screen_volume.py --ct case_ct.nii.gz --run runs/supcon --seg-ckpt runs/unet/best.pt --out screening/
   (or --mask some_mask.nii.gz to reuse an existing mask instead of the U-Net)"""
import argparse, pathlib, pandas as pd, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vertfrac.infer import screen_volume, load_classifier, load_cfg

ap = argparse.ArgumentParser(); ap.add_argument("--ct", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--seg-ckpt"); ap.add_argument("--mask"); ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--out", default="screening")
a = ap.parse_args(); cfg = load_cfg(a.config); dev = "cuda" if torch.cuda.is_available() else "cpu"
res, _, _ = screen_volume(a.ct, cfg, load_classifier(a.run, cfg, dev), mask_path=a.mask, seg_ckpt=a.seg_ckpt, device=dev)
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
pd.DataFrame([{k: v for k, v in r.items() if k not in ("roi", "heatmap")} for r in res]).to_csv(out / "screening.csv", index=False)
n = max(len(res), 1); fig, ax = plt.subplots(2, n, figsize=(2.3 * n, 5), squeeze=False)
for j, r in enumerate(res):
    ax[0, j].imshow(r["roi"][1], cmap="gray"); ax[0, j].set_title(f"V{r['vertebra']}: {r['grade_name']}\n{r['confidence']:.2f}", fontsize=8)
    ax[1, j].imshow(r["roi"][1], cmap="gray"); ax[1, j].imshow(r["heatmap"], cmap="jet", alpha=0.4)
    ax[0, j].axis("off"); ax[1, j].axis("off")
fig.suptitle("Decision support only - not a diagnosis", fontsize=9); fig.tight_layout(); fig.savefig(out / "overlay.png", dpi=150)
print(pd.read_csv(out / "screening.csv")[["vertebra", "grade_name", "confidence", "fractured_p"]].to_string(index=False))
