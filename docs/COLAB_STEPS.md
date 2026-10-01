# Run the project in Google Colab (backbone: Hugging Face microsoft/resnet-50)

Runtime -> Change runtime type -> **T4 GPU**. Paste each block into its own cell.

## 0. Get the code
```python
!git clone https://github.com/weasley29/vertfrac.git   # or upload the zip and !unzip it
%cd vertfrac
!pip install -q -r requirements.txt && pip install -q -e .

pytest -q tests          # all tests should pass (torch tests run on Colab)
```
(Optional) mount Drive so data survives runtime resets: `from google.colab import drive; drive.mount('/content/drive')`
and symlink `data/` into Drive.

## 1. Data
```python
!bash scripts/01_download_verse.sh        # six VerSe zips (~30-40 GB; use Drive or a bigger disk)
python scripts/02_inspect_dataset.py
```
Then get the **Genant grades**: open https://osf.io/nqjyw/ (Loffler et al. 2020, VerSe'19), download the
metadata/fracture-grade file into `data/`, and run:
```python
python scripts/02b_build_grades_csv.py --file data/<that file>
```
Check that the printed layout and counts look right (paper: 160 scans / 141 patients / 1,725 vertebrae).
Grades exist for VerSe'19 only, so VerSe'20 scans get grade -1 and are ignored by the classifier.

## 2. Crop vertebrae
```python
python scripts/03_extract_rois.py --limit 5     # quick test
python scripts/03_extract_rois.py               # all scans
```
Look at ~20 crops (matplotlib `plt.imshow(np.load(path)[1], cmap="gray")`): vertebra centred? upright? bone visible?

## 3. Train (backbone is set in configs/default.yaml -> hf:microsoft/resnet-50)
```python
python scripts/04_train_classifier.py --lam 0.0 --out runs/baseline   # cross-entropy only
python scripts/04_train_classifier.py --lam 0.5 --out runs/supcon      # + SupCon
```
Compare `runs/baseline/results.json` and `runs/supcon/results.json` (macro-F1, sensitivity, specificity, AUC,
confusion matrix). Repeat with seeds 0,1,2 (`--seed`).

## 4. Explain
```python
python scripts/05_gradcam_demo.py --run runs/supcon
```

## 5. Segmentation (after 1-4 work)
`python scripts/06_train_unet.py` (your own U-Net) and/or `scripts/09_hf_segment.py` (pretrained nnU-Net baseline).
