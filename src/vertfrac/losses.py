import torch, torch.nn as nn, torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al. 2020; Eq. 8 of the survey).
    z: (N,D) L2-normalised embeddings, y: (N,) labels. Anchors with no positive are skipped."""
    def __init__(self, temperature=0.1):
        super().__init__(); self.t = temperature

    def forward(self, z, y):
        n = z.size(0)
        sim = z @ z.T / self.t
        self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(self_mask, float("-inf"))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos = (y[:, None] == y[None, :]) & ~self_mask
        n_pos = pos.sum(1)
        valid = n_pos > 0
        if not valid.any():
            return z.sum() * 0.0
        mean_log_pos = log_prob.masked_fill(~pos, 0.0).sum(1)[valid] / n_pos[valid]
        return -mean_log_pos.mean()


class FocalCE(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__(); self.g, self.w = gamma, weight

    def forward(self, logits, y):
        logp = F.log_softmax(logits, 1)
        ce = F.nll_loss(logp, y, weight=self.w, reduction="none")
        pt = logp.gather(1, y[:, None]).squeeze(1).exp()
        return ((1 - pt) ** self.g * ce).mean()


def make_ce(class_counts=None, gamma=0.0, device="cpu"):
    w = None
    if class_counts is not None:
        c = torch.tensor(class_counts, dtype=torch.float32).clamp(min=1)
        w = (c.sum() / (len(c) * c)).to(device)          # inverse frequency
    return FocalCE(gamma, w) if gamma > 0 else nn.CrossEntropyLoss(weight=w)


class CombinedLoss(nn.Module):
    """L = L_CE + lambda * L_SupCon (Eq. 5). lambda=0 -> cross-entropy baseline."""
    def __init__(self, ce, lam=0.5, temperature=0.1):
        super().__init__(); self.ce, self.lam, self.sc = ce, lam, SupConLoss(temperature)

    def forward(self, logits, z, y_logits, y_z=None):
        l_ce = self.ce(logits, y_logits)
        l_sc = self.sc(z, y_z if y_z is not None else y_logits) if self.lam > 0 else torch.zeros((), device=z.device)
        return l_ce + self.lam * l_sc, {"ce": float(l_ce), "supcon": float(l_sc)}
