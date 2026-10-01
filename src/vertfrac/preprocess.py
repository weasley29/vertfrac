"""CT preprocessing + vertebra ROI extraction. Numpy/scipy/nibabel only (no torch)."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi

# VerSe label convention
VERT_NAMES = {**{i: f"C{i}" for i in range(1, 8)},
              **{7 + i: f"T{i}" for i in range(1, 13)},
              **{19 + i: f"L{i}" for i in range(1, 7)},
              28: "T13"}
THORACOLUMBAR = set(range(8, 26)) | {28}


def window_normalize(hu: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Eq. (2) in the survey paper: clip to window and rescale to [0, 1]."""
    return (np.clip(hu, hu_min, hu_max) - hu_min) / float(hu_max - hu_min)


def load_ct_and_mask(ct_path, mask_path, spacing_mm=1.0):
    """Load NIfTI, reorient to RAS, resample to isotropic spacing.
    Returns (ct_hu float32, mask int16) with axes (x=L-R, y=A-P, z=I-S)."""
    import nibabel as nib
    ct = nib.as_closest_canonical(nib.load(str(ct_path)))
    mk = nib.as_closest_canonical(nib.load(str(mask_path)))
    ct_arr = np.asarray(ct.dataobj, dtype=np.float32)   # NIfTI CT is already in HU
    mk_arr = np.asarray(mk.dataobj).astype(np.int16)
    zoom = np.array(ct.header.get_zooms()[:3]) / spacing_mm
    ct_arr = ndi.zoom(ct_arr, zoom, order=1)
    mk_arr = ndi.zoom(mk_arr, zoom, order=0)
    return ct_arr, mk_arr


def _crop_pad(img2d: np.ndarray, cy: int, cz: int, size: int, fill=0.0) -> np.ndarray:
    half = size // 2
    out = np.full((size, size), fill, dtype=img2d.dtype)
    y0, z0 = cy - half, cz - half
    ys, ye = max(y0, 0), min(y0 + size, img2d.shape[0])
    zs, ze = max(z0, 0), min(z0 + size, img2d.shape[1])
    if ye > ys and ze > zs:
        out[ys - y0:ye - y0, zs - z0:ze - z0] = img2d[ys:ye, zs:ze]
    return out


def extract_vertebra_roi(ct_hu, mask, label, *, hu_min, hu_max, roi_size=96,
                         offsets_mm=(-3, 0, 3), spacing_mm=1.0):
    """2.5D sagittal ROI around one vertebra -> float32 array (len(offsets), roi_size, roi_size),
    or None if the label is absent. Neighbouring vertebrae are NOT masked out by default
    (the classifier can use context, like a radiologist reading the film)."""
    vox = np.argwhere(mask == label)
    if vox.size == 0:
        return None
    cx, cy, cz = np.round(vox.mean(0)).astype(int)
    chans = []
    for off in offsets_mm:
        x = int(np.clip(cx + round(off / spacing_mm), 0, ct_hu.shape[0] - 1))
        sl = window_normalize(ct_hu[x], hu_min, hu_max)          # (y, z)
        chans.append(_crop_pad(sl, cy, cz, roi_size))
    roi = np.stack(chans, 0)
    return np.rot90(roi, k=1, axes=(1, 2)).copy()               # z up, display-friendly
