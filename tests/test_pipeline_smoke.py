"""End-to-end plumbing test on SYNTHETIC random volumes (no real data, numbers are meaningless):
ROI extraction -> manifest -> 2-epoch training -> tables. Takes ~1 minute on CPU."""
import json, pathlib, subprocess, sys, csv, numpy as np, pytest, yaml
nib = pytest.importorskip("nibabel"); pytest.importorskip("torch")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _fake_dataset(base):
    rng = np.random.default_rng(0); root = base / "data/verse/x"; rows = []
    for i in range(1, 31):
        s = f"sub-verse{i:03d}"; (root / "rawdata" / s).mkdir(parents=True); (root / "derivatives" / s).mkdir(parents=True)
        ct = rng.normal(-50, 40, (50, 50, 150)).astype(np.float32); mk = np.zeros(ct.shape, np.int16)
        for k, l in enumerate(range(8, 14)):
            z0, g = 10 + k * 22, int(rng.choice(4, p=[.5, .2, .2, .1])); h = [20, 18, 15, 11][g]
            ct[10:38, 10:38, z0:z0 + h] += 500; mk[10:38, 10:38, z0:z0 + h] = l; rows.append((s, l, g))
        aff = np.diag([1.5, 1.5, 1.5, 1.])
        nib.save(nib.Nifti1Image(ct, aff), root / "rawdata" / s / f"{s}_dir-ax_ct.nii.gz")
        nib.save(nib.Nifti1Image(mk, aff), root / "derivatives" / s / f"{s}_dir-ax_seg-vert_msk.nii.gz")
    with open(base / "data/grades.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject", "vert_label", "grade"]); w.writerows(rows)


def test_pipeline(tmp_path):
    (tmp_path / "configs").mkdir(); (tmp_path / "data").mkdir()
    (tmp_path / "configs/default.yaml").write_text((ROOT / "configs/default.yaml").read_text())
    _fake_dataset(tmp_path)
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"}
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    run(str(ROOT / "scripts/03_extract_rois.py"))
    assert (tmp_path / "data/manifest.csv").exists()
    run(str(ROOT / "scripts/04_train_classifier.py"), "--backbone", "resnet18", "--no-pretrained", "--epochs", "1",
        "--batch-size", "16", "--workers", "0", "--out", "runs/a/seed0")
    r = json.load(open(tmp_path / "runs/a/seed0/results.json"))
    assert r["config"]["backbone"] == "resnet18" and "macro_f1_4class" in r["test"]
    run(str(ROOT / "scripts/08_make_tables.py"), "--runs", "runs")
    assert (tmp_path / "results/ablation.csv").exists()
