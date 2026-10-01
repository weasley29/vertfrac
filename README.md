# vertfrac - Automatic Vertebrae Fracture Screening (chest CT)

VVCE final-year project. Pipeline: CT -> HU window -> vertebra segmentation (3D U-Net) -> per-vertebra 2.5D ROI ->
**Hugging Face `microsoft/resnet-50`** backbone -> Genant grade 0-3 with `L = CE + lambda*SupCon` -> Grad-CAM -> report.

## Status (be honest in your report)
- Verified on **synthetic** volumes only: all 10 unit/pipeline tests pass, and every script ran end-to-end
  (ROIs -> training -> ablation -> tables -> Grad-CAM -> U-Net -> end-to-end eval -> single-scan screening).
- **Not run on real VerSe data. No real results exist yet.** Numbers from the synthetic runs are meaningless.
- One manual step remains: the Genant grades are not in the VerSe zips. Get them from the VerSe'19 OSF project
  (https://osf.io/nqjyw/, Loffler et al. 2020) and run `scripts/02b_build_grades_csv.py` (written without seeing the real
  file - check its printed preview).

## Quick start
```
make install        # pip install -r requirements.txt && pip install -e .
make test
make data           # download VerSe (30-40 GB)
python scripts/02b_build_grades_csv.py --file data/<grades file>
make rois
make baseline && make supcon
make ablation && make tables
make unet
python scripts/10_eval_pred_masks.py --run runs/supcon --seg-ckpt runs/unet/best.pt
make app            # Streamlit demo
```
Colab: open `notebooks/colab_run.ipynb`.

## Layout
| Path | What |
|---|---|
| `src/vertfrac/` | preprocess, splits, dataset, models (HF ResNet-50 / torchvision / timm), losses (SupCon, focal), gradcam, metrics, instances, infer |
| `scripts/01-03` | download, inspect, grades csv (02b), ROI extraction + patient-level split |
| `scripts/04` | classifier training (`--lam 0` = baseline) |
| `scripts/05` | Grad-CAM figure |
| `scripts/06` | your 3D U-Net (MONAI), patient-level split, Dice on val/test |
| `scripts/07-08` | ablation grid runner, tables/figures (mean +/- std) |
| `scripts/09` | optional pretrained nnU-Net (HF `fhofmann/VertebralBodiesCT-ResEncM`); trained on data incl. VerSe -> not an unbiased baseline |
| `scripts/10` | end-to-end evaluation with predicted masks (GT-mask vs predicted-mask gap) |
| `scripts/11`, `app.py` | screen one CT: CSV + overlay / Streamlit UI |
| `docs/ROADMAP.md`, `docs/COLAB_STEPS.md` | plan and Colab steps |

## Design decisions (say these in your viva)
- Split by **patient**, stratified on grade; model selection on val only; test evaluated once.
- No mirror-flip augmentation (sagittal anatomy is not left-right symmetric).
- Imbalance handled by balanced sampling + class-weighted CE (+ optional focal loss); SupCon is label-based.
- 2.5D input: three sagittal slices (-3, 0, +3 mm) as RGB-like channels for the pretrained backbone.
- Vertebra numbering from a binary mask is top-to-bottom (`instances.py`); level names (T1..L5) need a labelling model.
- Always report per-grade counts with every metric.

## Known limits
Thoracolumbar only (VerSe grades exclude cervical). Only VerSe'19 has Genant grades (160 scans, 1,725 vertebrae; ~51 severe).
Public grades have rater variability; grade 0 vs 1 will be the hardest boundary. `instances.py` cannot split vertebrae fused
into a solid column. No external validation yet. Research prototype - not a diagnostic device.
