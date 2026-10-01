"""Optional: segment vertebrae with a pretrained nnU-Net from Hugging Face (fhofmann/VertebralBodiesCT-ResEncM).
UNTESTED helper - needs nnunetv2 + a GPU with ~10 GB VRAM (Colab T4 works). Steps:
  1. downloads the model folder from the Hub
  2. runs nnUNetv2_predict_from_modelfolder on a folder of *_0000.nii.gz CTs
IMPORTANT: that model was trained on public data that includes VerSe scans, so do NOT report its VerSe-test
Dice as an unbiased result. Use it as a convenience/baseline, and read its dataset.json for the label ids
(the ids may differ from VerSe's T1..L5 numbering) before feeding masks to 03_extract_rois.py."""
import argparse, os, pathlib, subprocess
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True, help="folder with case_0000.nii.gz files")
ap.add_argument("--out", required=True)
ap.add_argument("--repo", default="fhofmann/VertebralBodiesCT-ResEncM")
a = ap.parse_args()
root = pathlib.Path(snapshot_download(a.repo))
model_dir = next(root.glob("nnUNetTrainer__*__3d_fullres"))
print("model folder:", model_dir, "| dataset.json:", (model_dir / "dataset.json").exists())
subprocess.run(["nnUNetv2_predict_from_modelfolder", "-i", a.inp, "-o", a.out, "-m", str(model_dir), "-f", "all"], check=True)
