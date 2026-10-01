"""Step 3 - train the Genant-grade classifier (CE, or CE + lambda*SupCon).
Baseline:  python scripts/04_train_classifier.py --lam 0.0 --out runs/baseline
SupCon:    python scripts/04_train_classifier.py --lam 0.5 --out runs/supcon
Every run writes results.json (config + val curve + TEST metrics), best.pt, test_probs.npz.
Model selection uses the VAL split only; the TEST split is evaluated once at the end."""
import argparse, json, pathlib, time, yaml, numpy as np, torch
from torch.utils.data import DataLoader
from vertfrac.dataset import VertebraROIDataset
from vertfrac.models import FractureNet
from vertfrac.losses import CombinedLoss, make_ce
from vertfrac.metrics import grade_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--out")
ap.add_argument("--lam", type=float); ap.add_argument("--temp", type=float); ap.add_argument("--seed", type=int)
ap.add_argument("--backbone"); ap.add_argument("--epochs", type=int); ap.add_argument("--batch-size", type=int)
ap.add_argument("--focal-gamma", type=float); ap.add_argument("--balanced", type=int, choices=[0, 1])
ap.add_argument("--no-pretrained", action="store_true", help="random init (offline smoke tests)")
ap.add_argument("--amp", action="store_true", help="mixed precision on GPU"); ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); t, m = cfg["train"], cfg["model"]
lam = t["lambda_supcon"] if a.lam is None else a.lam
temp = t["temperature"] if a.temp is None else a.temp
seed = t["seed"] if a.seed is None else a.seed
backbone = a.backbone or m["backbone"]; epochs = a.epochs or t["epochs"]; bs = a.batch_size or t["batch_size"]
gamma = t["focal_gamma"] if a.focal_gamma is None else a.focal_gamma
balanced = bool(t["balanced_sampling"]) if a.balanced is None else bool(a.balanced)
out = pathlib.Path(a.out or t["out_dir"]); out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(seed); np.random.seed(seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

man = cfg["data"]["manifest"]
tr = VertebraROIDataset(man, "train", train=True, two_views=True)
va = VertebraROIDataset(man, "val"); te = VertebraROIDataset(man, "test")
print("device:", dev, "| grades train/val/test:", tr.class_counts().tolist(), va.class_counts().tolist(), te.class_counts().tolist())
assert len(tr) > 0 and len(va) > 0 and len(te) > 0, "empty split - check grades.csv / manifest (grade -1 rows are dropped)"
sampler = tr.balanced_sampler() if balanced else None
tl = DataLoader(tr, bs, sampler=sampler, shuffle=sampler is None, num_workers=a.workers, drop_last=len(tr) > bs)

net = FractureNet(backbone, not a.no_pretrained and m["pretrained"], m["embed_dim"], m["num_classes"]).to(dev)
ce = make_ce(tr.class_counts() if t["class_weighted_ce"] else None, gamma, dev)
crit = CombinedLoss(ce, lam, temp)
opt = torch.optim.AdamW(net.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
scaler = torch.amp.GradScaler("cuda", enabled=a.amp and dev == "cuda")


@torch.no_grad()
def predict(ds):
    net.eval(); P, Y = [], []
    for x, y in DataLoader(ds, 64):
        P.append(torch.softmax(net(x.to(dev))[0].float(), 1).cpu()); Y.append(y)
    return torch.cat(Y).numpy(), torch.cat(P).numpy()


best, log, t0 = -1.0, [], time.time()
for ep in range(epochs):
    net.train(); run = []
    for x, y in tl:                                   # x: (B,2,C,H,W) two augmented views
        x = x.flatten(0, 1).to(dev); y2 = y.to(dev).repeat_interleave(2)
        with torch.autocast(dev, enabled=a.amp and dev == "cuda"):
            logits, z = net(x)
        loss, parts = crit(logits.float(), z.float(), y2)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); run.append(float(loss))
    sched.step()
    yv, pv = predict(va); mv = grade_metrics(yv, pv)
    log.append(dict(epoch=ep, train_loss=float(np.mean(run)), val_macro_f1=mv["macro_f1_4class"],
                    val_auc=mv["auc_binary"], val_sens=mv["sensitivity"], val_spec=mv["specificity"]))
    print(log[-1])
    if mv["macro_f1_4class"] > best:                  # selection on VAL only
        best = mv["macro_f1_4class"]; torch.save(net.state_dict(), out / "best.pt")

net.load_state_dict(torch.load(out / "best.pt", map_location=dev, weights_only=True)); yt, pt = predict(te)
config = dict(lam=lam, temperature=temp, backbone=backbone, focal_gamma=gamma, balanced=balanced, seed=seed,
              epochs=epochs, batch_size=bs, lr=t["lr"], pretrained=not a.no_pretrained and m["pretrained"])
json.dump(dict(config=config, best_val_macro_f1=best, seconds=round(time.time() - t0, 1),
               test=grade_metrics(yt, pt), log=log), open(out / "results.json", "w"), indent=2)
np.savez(out / "test_probs.npz", y=yt, p=pt)
print("saved", out / "results.json")
