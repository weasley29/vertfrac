"""Streamlit demo:  streamlit run app.py
Upload a CT (.nii.gz) -> segmentation (your U-Net checkpoint) -> per-vertebra Genant grade, confidence, Grad-CAM."""
import tempfile, pathlib, pandas as pd, streamlit as st, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vertfrac.infer import screen_volume, load_classifier, load_cfg

st.set_page_config(page_title="Vertebral fracture screening", layout="wide")
st.title("Automatic Vertebrae Fracture Screening")
st.warning("Research prototype - decision support only, not a diagnosis.")
cfg = load_cfg(); dev = "cuda" if torch.cuda.is_available() else "cpu"
run = st.sidebar.text_input("Classifier run folder", "runs/supcon")
seg = st.sidebar.text_input("U-Net checkpoint", "runs/unet/best.pt")
f = st.file_uploader("CT volume (.nii.gz)", type=["gz", "nii"])
if f and st.button("Screen"):
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "ct.nii.gz"; p.write_bytes(f.getvalue())
        with st.spinner("Segmenting and grading..."):
            res, _, _ = screen_volume(p, cfg, load_classifier(run, cfg, dev), seg_ckpt=seg, device=dev)
    if not res: st.error("No vertebrae found.")
    else:
        st.dataframe(pd.DataFrame([{k: v for k, v in r.items() if k not in ("roi", "heatmap", "centroid_xyz")} for r in res]))
        cols = st.columns(min(len(res), 6))
        for i, r in enumerate(res):
            fig, ax = plt.subplots(figsize=(2.4, 2.4)); ax.imshow(r["roi"][1], cmap="gray"); ax.imshow(r["heatmap"], cmap="jet", alpha=.4)
            ax.axis("off"); ax.set_title(f"V{r['vertebra']} {r['grade_name']} ({r['confidence']:.2f})", fontsize=8)
            cols[i % len(cols)].pyplot(fig); plt.close(fig)
