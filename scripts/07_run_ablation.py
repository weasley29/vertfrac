"""Runs the ablation grid by calling 04_train_classifier.py once per setting x seed.
   python scripts/07_run_ablation.py --quick     # tiny grid, 3 epochs: checks the plumbing
   python scripts/07_run_ablation.py              # full grid (many GPU hours: trim GRID to what you can afford)
Runs land in runs/ablation/<tag>/seed<k>/ ; resume-safe (skips finished runs). Then: python scripts/08_make_tables.py"""
import argparse, itertools, pathlib, subprocess, sys

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--root", default="runs/ablation")
ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="passed to 04 (e.g. --no-pretrained --workers 0)")
a = ap.parse_args()

GRID = dict(lam=[0.0, 0.1, 0.5, 1.0], temp=[0.1], backbone=["hf:microsoft/resnet-50", "efficientnet_b0"],
            gamma=[0.0], balanced=[1])
EXTRA_GRID = [dict(lam=0.5, temp=t) for t in (0.07, 0.2)] + [dict(lam=0.5, gamma=2.0), dict(lam=0.5, balanced=0)]
if a.quick:
    GRID = dict(lam=[0.0, 0.5], temp=[0.1], backbone=["resnet18"], gamma=[0.0], balanced=[1]); EXTRA_GRID = []; a.seeds = [0]

runs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
base = dict(lam=0.5, temp=0.1, backbone="hf:microsoft/resnet-50", gamma=0.0, balanced=1)
runs += [{**base, **e} for e in EXTRA_GRID]
seen, todo = set(), []
for r in runs:
    key = tuple(sorted(r.items()))
    if key not in seen: seen.add(key); todo.append(r)
for r in todo:
    tag = "_".join(f"{k}{str(v).replace(':', '-').replace('/', '-')}" for k, v in r.items())
    for s in a.seeds:
        out = pathlib.Path(a.root) / tag / f"seed{s}"
        if (out / "results.json").exists(): print("skip", out); continue
        cmd = [sys.executable, str(pathlib.Path(__file__).with_name("04_train_classifier.py")), "--out", str(out), "--seed", str(s),
               "--lam", str(r["lam"]), "--temp", str(r["temp"]), "--backbone", r["backbone"],
               "--focal-gamma", str(r["gamma"]), "--balanced", str(r["balanced"])]
        if a.quick: cmd += ["--epochs", "3"]
        cmd += a.extra; print(">>", " ".join(cmd)); subprocess.run(cmd, check=True)
