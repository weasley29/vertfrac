from __future__ import annotations
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torch.nn.functional as F

MEAN, STD = 0.45, 0.25   # rough; recompute from the train split later


def _augment(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """x: (C,H,W) in [0,1]. Deliberately NO mirror flip: sagittal anatomy is not left-right
    symmetric (anterior vs posterior), so flips can create unrealistic vertebrae.
    Small rotation + scale + shift + contrast/brightness + noise only."""
    ang = np.deg2rad(rng.uniform(-10, 10)); sc = rng.uniform(0.9, 1.1)
    c, s = np.cos(ang) / sc, np.sin(ang) / sc
    theta = torch.tensor([[c, -s, rng.uniform(-.05, .05)], [s, c, rng.uniform(-.05, .05)]],
                         dtype=torch.float32)[None]
    grid = F.affine_grid(theta, (1, *x.shape), align_corners=False)
    x = F.grid_sample(x[None], grid, padding_mode="border", align_corners=False)[0]
    x = (x - 0.5) * rng.uniform(0.85, 1.15) + 0.5 + rng.uniform(-0.05, 0.05)
    x = x + torch.randn_like(x) * rng.uniform(0, 0.02)
    return x.clamp(0, 1)


class VertebraROIDataset(Dataset):
    def __init__(self, manifest_csv, split, *, train=False, two_views=False, drop_unlabelled=True):
        df = pd.read_csv(manifest_csv)
        df = df[df.split == split]
        if drop_unlabelled:
            df = df[df.grade >= 0]
        self.df = df.reset_index(drop=True)
        self.train, self.two_views = train, two_views
        self.rng = np.random.default_rng(0)

    def __len__(self): return len(self.df)

    def _prep(self, arr, aug):
        x = torch.from_numpy(arr.astype(np.float32))
        if aug: x = _augment(x, self.rng)
        return (x - MEAN) / STD

    def __getitem__(self, i):
        r = self.df.iloc[i]
        arr = np.load(r.roi_path)
        y = int(r.grade)
        if self.two_views:                       # two augmented views for SupCon
            return torch.stack([self._prep(arr, True), self._prep(arr, True)]), y
        return self._prep(arr, self.train), y

    def class_counts(self, n=4):
        return np.bincount(self.df.grade.to_numpy(), minlength=n)

    def balanced_sampler(self):
        w = 1.0 / np.maximum(self.class_counts(), 1)
        return WeightedRandomSampler(w[self.df.grade.to_numpy()], num_samples=len(self.df), replacement=True)
