"""Step 4 - your own 3D U-Net (MONAI) for binary thoracolumbar-vertebra segmentation, loss = Dice + BCE (Eq. 3-4).
Uses the PATIENT-LEVEL split from data/manifest.csv (train/val/test), reports Dice on val (selection) and test.
Vertebra numbering is done afterwards by vertfrac.instances.label_instances (top->bottom, no level names).
   python scripts/06_train_unet.py            # full run (GPU)
   python scripts/06_train_unet.py --epochs 2 --patch 32 32 32 --spacing 3   # smoke test"""
import argparse, json, pathlib, re, yaml, numpy as np, pandas as pd, torch
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.data import CacheDataset, DataLoader
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged,
                              RandCropByPosNegLabeld, RandAffined, Lambdad, EnsureTyped)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--epochs", type=int); ap.add_argument("--patch", type=int, nargs=3); ap.add_argument("--spacing", type=float)
ap.add_argument("--val-every", type=int, default=5); ap.add_argument("--out"); ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); s, p = cfg["seg"], cfg["preprocess"]
epochs = a.epochs or s["epochs"]; patch = tuple(a.patch or s["patch_size"]); spacing = a.spacing or s["spacing_mm"]
root = pathlib.Path(cfg["data"]["verse_root"]); out = pathlib.Path(a.out or s["out_dir"]); out.mkdir(parents=True, exist_ok=True)
masks = {re.sub(r"_seg-vert_msk\.nii\.gz$", "", q.name): q for q in root.rglob("*_seg-vert_msk.nii.gz")}
split_of = pd.read_csv(cfg["data"]["manifest"]).drop_duplicates("subject").set_index("subject")["split"].to_dict()
items = {"train": [], "val": [], "test": []}
for c in sorted(root.rglob("*_ct.nii.gz")):
    k = re.sub(r"_ct\.nii\.gz$", "", c.name)
    if k in masks and k in split_of:
        items[split_of[k]].append(dict(image=str(c), label=str(masks[k])))
print({k: len(v) for k, v in items.items()}); assert items["train"] and items["val"], "run 03_extract_rois.py first"

sp = (spacing,) * 3
base = [LoadImaged(["image", "label"]), EnsureChannelFirstd(["image", "label"]), Orientationd(["image", "label"], "RAS"),
        Spacingd(["image", "label"], pixdim=sp, mode=("bilinear", "nearest")),
        ScaleIntensityRanged("image", p["hu_min"], p["hu_max"], 0.0, 1.0, clip=True),
        Lambdad("label", lambda x: (((x >= 8) & (x <= 25)) | (x == 28)).float())]      # thoracolumbar -> 1
tr_tf = Compose(base + [RandCropByPosNegLabeld(["image", "label"], "label", patch, pos=2, neg=1, num_samples=2),
                        RandAffined(["image", "label"], prob=0.3, rotate_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest")),
                        EnsureTyped(["image", "label"])])
va_tf = Compose(base + [EnsureTyped(["image", "label"])])
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = UNet(3, 1, 1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2).to(dev)
loss_fn = DiceCELoss(sigmoid=True); opt = torch.optim.AdamW(net.parameters(), 1e-3)
tl = DataLoader(CacheDataset(items["train"], tr_tf, cache_rate=0.1, num_workers=0), batch_size=s["batch_size"], shuffle=True, num_workers=a.workers)
dice = DiceMetric(include_background=True)


def evaluate(split):
    net.eval(); ds = DataLoader(CacheDataset(items[split], va_tf, cache_rate=0.0, num_workers=0), batch_size=1)
    with torch.no_grad():
        for b in ds:
            pr = (torch.sigmoid(sliding_window_inference(b["image"].to(dev), patch, 2, net)) > 0.5).float()
            dice(pr, b["label"].to(dev))
    d = float(dice.aggregate().item()); dice.reset(); return d


best, log = -1.0, []
for ep in range(epochs):
    net.train(); run = []
    for b in tl:
        opt.zero_grad(); loss = loss_fn(net(b["image"].to(dev)), b["label"].to(dev)); loss.backward(); opt.step(); run.append(float(loss))
    rec = dict(epoch=ep, train_loss=float(np.mean(run)))
    if (ep + 1) % a.val_every == 0 or ep == epochs - 1:
        rec["val_dice"] = evaluate("val"); print(rec)
        if rec["val_dice"] > best: best = rec["val_dice"]; torch.save(net.state_dict(), out / "best.pt")
    log.append(rec)
net.load_state_dict(torch.load(out / "best.pt", map_location=dev, weights_only=True))
res = dict(best_val_dice=best, test_dice=evaluate("test") if items["test"] else None, spacing=spacing, patch=patch, log=log)
json.dump(res, open(out / "results.json", "w"), indent=2); print({k: v for k, v in res.items() if k != "log"})
