.PHONY: install test data rois baseline supcon ablation tables unet app
install: ; pip install -r requirements.txt && pip install -e .
test:    ; pytest -q tests
data:    ; bash scripts/01_download_verse.sh && python scripts/02_inspect_dataset.py
rois:    ; python scripts/03_extract_rois.py
baseline:; python scripts/04_train_classifier.py --lam 0.0 --out runs/baseline --amp
supcon:  ; python scripts/04_train_classifier.py --lam 0.5 --out runs/supcon --amp
ablation:; python scripts/07_run_ablation.py --extra --amp
tables:  ; python scripts/08_make_tables.py
unet:    ; python scripts/06_train_unet.py
app:     ; streamlit run app.py
