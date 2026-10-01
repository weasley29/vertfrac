"""End-to-end screening of one CT volume: segment -> instances -> ROI -> grade + Grad-CAM."""
from __future__ import annotations
import numpy as np, torch, yaml
from scipy import ndimage as ndi
from .preprocess import load_ct_and_mask, extract_vertebra_roi, window_normalize
from .instances import label_instances
from .models import FractureNet
from .dataset import MEAN, STD
from .gradcam import GradCAM

GRADE_NAMES = {0: "normal", 1: "mild", 2: "moderate", 3: "severe"}


def load_classifier(run_dir, cfg, device="cpu"):
    import json, pathlib
    m = dict(cfg["model"]); rj = pathlib.Path(run_dir) / "results.json"
    if rj.exists():                                   # use the backbone this run was actually trained with
        m["backbone"] = json.load(open(rj))["config"]["backbone"]
    net = FractureNet(m["backbone"], pretrained=False, embed_dim=m["embed_dim"], num_classes=m["num_classes"])
    net.load_state_dict(torch.load(f"{run_dir}/best.pt", map_location=device, weights_only=True))
    return net.to(device).eval()


@torch.no_grad()
def predict_mask_unet(ct_hu_1mm: np.ndarray, ckpt: str, cfg: dict, device="cpu") -> np.ndarray:
    """Binary vertebra mask on the SAME grid as ct_hu_1mm, from the MONAI U-Net trained by 06_train_unet.py."""
    from monai.networks.nets import UNet
    from monai.inferers import sliding_window_inference
    s, p = cfg["seg"], cfg["preprocess"]
    factor = p["spacing_mm"] / s["spacing_mm"]
    small = ndi.zoom(ct_hu_1mm, factor, order=1)
    x = torch.from_numpy(window_normalize(small, p["hu_min"], p["hu_max"]).astype(np.float32))[None, None].to(device)
    net = UNet(3, 1, 1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)); net.eval()
    pr = (torch.sigmoid(sliding_window_inference(x, tuple(s["patch_size"]), 2, net)) > 0.5)[0, 0].cpu().numpy()
    zoom = np.array(ct_hu_1mm.shape) / np.array(pr.shape)
    return ndi.zoom(pr.astype(np.uint8), zoom, order=0)[tuple(slice(0, n) for n in ct_hu_1mm.shape)]


def screen_volume(ct_path, cfg, classifier, *, mask_path=None, seg_ckpt=None, device="cpu", with_cam=True):
    """Returns (list of per-vertebra dicts, ct_hu, instance_mask). Provide mask_path (an existing
    binary/instance mask, e.g. from nnU-Net) OR seg_ckpt (your U-Net weights)."""
    p = cfg["preprocess"]
    if mask_path is not None:
        ct_hu, mk = load_ct_and_mask(ct_path, mask_path, p["spacing_mm"])
    else:
        assert seg_ckpt, "need mask_path or seg_ckpt"
        ct_hu, _ = load_ct_and_mask(ct_path, ct_path, p["spacing_mm"])     # mask arg unused; reuse loader for the CT
        mk = predict_mask_unet(ct_hu, seg_ckpt, cfg, device)
    inst = label_instances(mk)
    cam = GradCAM(classifier, classifier.gradcam_layer()) if with_cam else None
    out = []
    for lab in [int(l) for l in np.unique(inst) if l != 0]:
        roi = extract_vertebra_roi(ct_hu, inst, lab, hu_min=p["hu_min"], hu_max=p["hu_max"], roi_size=p["roi_size"],
                                   offsets_mm=p["slice_offsets_mm"], spacing_mm=p["spacing_mm"])
        if roi is None:
            continue
        x = ((torch.from_numpy(roi.astype(np.float32)) - MEAN) / STD)[None].to(device)
        if cam is not None:
            heat, logits = cam(x)
            heat = heat[0, 0].cpu().numpy()
        else:
            with torch.no_grad():
                logits = classifier(x)[0]
            heat = None
        prob = torch.softmax(logits, 1)[0].cpu().numpy()
        g = int(prob.argmax())
        out.append(dict(vertebra=lab, order="top->bottom", grade=g, grade_name=GRADE_NAMES[g], confidence=float(prob[g]),
                        p_normal=float(prob[0]), p_mild=float(prob[1]), p_moderate=float(prob[2]), p_severe=float(prob[3]),
                        fractured_p=float(1 - prob[0]), centroid_xyz=np.argwhere(inst == lab).mean(0).round(1).tolist(),
                        roi=roi, heatmap=heat))
    return out, ct_hu, inst


def load_cfg(path="configs/default.yaml"):
    return yaml.safe_load(open(path))
