"""Grad-CAM panel for the Phase-2 'Demo 2' slide: ROI, predicted grade + confidence, heatmap."""
import argparse, yaml, numpy as np, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vertfrac.dataset import VertebraROIDataset
from vertfrac.infer import load_classifier
from vertfrac.gradcam import GradCAM

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=8)
ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--out", default="gradcam_demo.png")
a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); m = cfg["model"]
net = load_classifier(a.run, cfg)
cam = GradCAM(net, net.gradcam_layer()); ds = VertebraROIDataset(cfg["data"]["manifest"], "test")
idx = np.random.default_rng(0).choice(len(ds), min(a.n, len(ds)), replace=False)
fig, ax = plt.subplots(2, len(idx), figsize=(2.4 * len(idx), 5))
for j, i in enumerate(idx):
    x, y = ds[i]; heat, logits = cam(x[None]); p = torch.softmax(logits, 1)[0]
    img = x[1].numpy(); ax[0, j].imshow(img, cmap="gray"); ax[0, j].set_title(f"true G{y} / pred G{int(p.argmax())}\n{float(p.max()):.2f}", fontsize=8)
    ax[1, j].imshow(img, cmap="gray"); ax[1, j].imshow(heat[0, 0], cmap="jet", alpha=0.4)
    for r in (0, 1): ax[r, j].axis("off")
plt.tight_layout(); plt.savefig(a.out, dpi=150); print("saved", a.out)
