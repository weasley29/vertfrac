"""Split a binary 'any vertebra' mask into one label per vertebra (no anatomical level names).
Approach: distance-transform cores (voxels that are near the local maximum thickness, so thin
disc-level connections drop out) -> connected components -> grow labels back inside the mask ->
merge fragments of the same vertebra (body + posterior arch share almost the same z) -> number
top->bottom. Level names (T1..L5) are NOT known from a binary mask; use a labelling model if you need them.
Limits: vertebrae fused into a solid column with no narrowing cannot be separated by any shape rule."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


def label_instances(binary_mask: np.ndarray, frac: float = 0.7, radius: int = 12, min_voxels: int = 300,
                    min_z_sep: float = 10.0) -> np.ndarray:
    """binary_mask (x,y,z), z = inferior->superior (RAS), voxel units (use ~1 mm spacing).
    Returns int16: 0 = background, 1..n = vertebrae ordered from the top of the volume downwards."""
    m_full = binary_mask > 0
    out = np.zeros(m_full.shape, np.int16)
    if not m_full.any():
        return out
    idx = np.argwhere(m_full); lo, hi = idx.min(0), idx.max(0) + 1              # work on the bounding box only
    box = tuple(slice(a, b) for a, b in zip(lo, hi)); m = m_full[box]
    dist = ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1, 1:-1]
    local_max = ndi.maximum_filter(dist, size=2 * radius + 1)
    core = m & (dist >= frac * local_max) & (dist >= 2)
    seeds, n = ndi.label(core)
    if n == 0:
        return out
    _, near = ndi.distance_transform_edt(seeds == 0, return_indices=True)     # nearest seed for every voxel
    grown = seeds[tuple(near)] * m
    ids = [i for i in np.unique(grown) if i != 0]
    zc = {i: float(np.argwhere(grown == i)[:, 2].mean()) for i in ids}
    order = sorted(ids, key=lambda k: -zc[k])
    groups, cur = [], [order[0]]
    for i in order[1:]:                                                        # merge fragments at the same height
        if abs(zc[cur[-1]] - zc[i]) < min_z_sep:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    rank = 0
    for g in groups:
        sel = np.isin(grown, g)
        if sel.sum() < min_voxels:
            continue
        rank += 1
        out[box][sel] = rank
    return out
