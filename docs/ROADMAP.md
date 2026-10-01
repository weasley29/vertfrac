# Next Steps: Automatic Vertebrae Fracture Screening (VVCE final-year project)

Status now: synopsis + survey paper done, Phase 2 slides drafted, no experiments run.
Goal of this plan: get from "design" to "first honest numbers" and fill the empty slots in
your Phase 2 deck (Demo 1, Demo 2, Algorithms Tried, Results, Paper Status).

Key idea: **the classifier does not need your U-Net.** VerSe ships ground-truth vertebra masks, so
build the classifier on GT-mask crops first (Stage 2), then add your own segmentation (Stage 4) and
measure how much accuracy drops when GT masks are replaced by predicted ones. That order removes
the biggest schedule risk.

## Step 0 - Fix these before you build
Found while cross-reading your files; worth correcting in the survey and synopsis.
1. **Ref [6] is mis-attributed.** The VerSe fracture-grading paper is Loffler M. et al., "A Vertebral
   Segmentation Dataset with Fracture Grading", Radiology: AI, **2020** (doi 10.1148/ryai.2020190138),
   not "Sollmann ... 2021". Also fix its year in Table I.
2. **Class counts belong to a subset.** 1,436 / 142 / 96 / 51 vertebrae is reported for **VerSe19
   (160 scans, 141 patients)** in other papers, not for all 374 scans. Rewrite Sec. VII-A and Sec. X
   so they do not say "374 scans with ... 1,436 normal ...". Confirm the real numbers yourself in step 1.3.
3. **Ref [1] venue differs** between synopsis (CVPR Workshops) and survey (IEEE BIBM). Check the
   paper's own first page and use one.
4. **RSNA 2022 (in your dataset_link file) is cervical (C1-C7) with binary fracture labels**, so it
   cannot train a thoracolumbar Genant grader. Use it only as optional external binary test, or drop it.
   CTSpine1K has no fracture labels: segmentation pretraining only.

## Step 1 - Data
1.1 `bash scripts/01_download_verse.sh` (six zips from your Pages file; ~30-40 GB, use a machine or Drive with space).
1.2 `python scripts/02_inspect_dataset.py`: confirms CT/mask pairs, subject ids, repeat scans, grade files.
1.3 **Get the Genant grades (they are NOT in the s3 zips).** The zips contain only CT, mask, centroid json and a PNG. Per-vertebra Genant grades come with Loffler et al. 2020 for **VerSe'19 only** (160 scans, 141 patients, 1,725 vertebrae) via the OSF project https://osf.io/nqjyw/. Download the metadata file there, then run `python scripts/02b_build_grades_csv.py --file <that file>` (check its printed preview).
    Old note: Read the VerSe GitHub README and the Loffler paper for where per-vertebra
    grades live, then write `data/grades.csv` with columns `subject,vert_label,grade` (grade 0-3).
    Only a subset of scans carries grades; ungraded vertebrae are kept as `-1` and excluded from
    classifier training. Report the true per-grade counts in your paper.
1.4 `python scripts/03_extract_rois.py --limit 5` then full run. Look at 20 ROIs by eye (matplotlib) before trusting them.
    Check orientation, that the vertebra is centred, and that the bone window looks right.
**Done when:** `data/manifest.csv` exists, no patient in two splits, per-grade table printed.

## Step 2 - Baseline classifier
2.1 `python scripts/04_train_classifier.py --lam 0.0 --out runs/baseline` (cross-entropy only).
2.2 Sanity checks first: overfit 32 samples; shuffle labels (metrics must collapse to chance); confirm val/test counts per grade.
2.3 Tune only on **val**. Touch the test fold once, at the end.
**Done when:** baseline results.json with macro-F1, sensitivity, specificity, AUC, confusion matrix.
(Phase 2 slide: "Algorithms Tried" = ResNet18 CE baseline, class-weighted CE, balanced sampling.)

## Step 3 - SupCon + ablation
3.1 `python scripts/04_train_classifier.py --lam 0.5 --out runs/supcon`.
3.2 Ablation grid (each with 3 seeds, report mean +/- std; with ~50 severe vertebrae one seed is noise):
    lambda in {0, 0.1, 0.5, 1.0}; temperature {0.07, 0.1, 0.2}; backbone {resnet18, efficientnet_b0};
    focal gamma {0, 2}; balanced sampling on/off.
3.3 Report per-grade F1, especially **grade 0 vs 1**, since that is your paper's central claim.
    If SupCon does not help, that is still a publishable negative result; say so.
3.4 `python scripts/05_gradcam_demo.py --run runs/supcon` for the "Demo 2" slide. Include failure cases too.
**Done when:** ablation table + Grad-CAM panel.

## Step 4 - Segmentation
4.1 `python scripts/06_train_unet.py` (MONAI 3D U-Net, Dice+BCE, binary foreground). Metric: Dice on val.
4.2 Extra baseline for free: run TotalSegmentator / nnU-Net vertebra models as a comparison.
4.3 Labelling T1..L5 (instance identification) is harder than binary segmentation. Options: connected
    components + counting from a known landmark, a centroid/label head, or use a pretrained labeller.
4.4 Rerun ROI extraction from **predicted** masks, re-evaluate the classifier without retraining, and
    report the GT-mask vs predicted-mask gap. That is the honest end-to-end number.
(Phase 2 slide: "Demo 1" = CT slice -> U-Net mask -> labelled vertebrae.)

## Step 5 - End-to-end app
Input NIfTI -> preprocess -> segment -> per-vertebra grade + confidence + Grad-CAM overlay -> table/PNG.
A Streamlit or Gradio page is enough. Show "decision support, not diagnosis" on screen.

## Step 6 - Evaluation, paper, report
- Patient-level 5-fold CV (change `test_fold`) for final numbers with confidence intervals.
- Optional external check: RSNA cervical binary (label mismatch: say so) or a small held-out VerSe20 set.
- Update the survey into a full paper: replace "no results" sections with your tables; keep Sec. X honest.
- Freeze env (`pip freeze`), seeds, configs, and commit them.

## Risks
| Risk | Mitigation |
|---|---|
| Grade files hard to locate/parse | Do step 1.3 on day 1; email VerSe maintainers (address in their README) |
| Only ~50 severe vertebrae | Report per-grade counts; merge grades 2+3 as a secondary analysis; CV + 3 seeds |
| Label noise around grade 0/1 | Report confusion matrix, not just accuracy |
| No GPU | Colab; ResNet18 at 96x96 is light; U-Net needs more (crop patches, 1.5 mm spacing) |
| Leakage inflates results | `assert_no_leakage` runs in step 03; never split by vertebra |
| Time | If behind: skip your own U-Net (use TotalSegmentator) and spend time on the ablation |

## Phase 2 deck: what fills each empty slot
| Slide | Source |
|---|---|
| Algorithms Tried | Stage 2-3 ablation rows |
| Implementation & Demo 1 | Stage 4 screenshot (or GT-mask ROI montage if U-Net not ready) |
| Implementation & Demo 2 | `05_gradcam_demo.py` output |
| Results & challenges | `runs/*/results.json` |
| Paper status | fill journal/conference + status |
