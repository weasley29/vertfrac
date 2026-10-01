import torch.nn.functional as F


class GradCAM:
    """Eq. (9)-(10): alpha_k = GAP(dO_c/dA_k); M = ReLU(sum_k alpha_k A_k), upsampled to input size.
    Uses a forward hook + tensor hook (robust to in-place ReLUs in torchvision/timm ResNets)."""
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        def fwd(m, i, o):
            self.acts = o
            if o.requires_grad:
                o.register_hook(lambda g: setattr(self, "grads", g))
        layer.register_forward_hook(fwd)

    def __call__(self, x, class_idx=None):
        self.model.eval(); self.model.zero_grad()
        x = x.clone().requires_grad_(True)                # ensures a graph even if weights are frozen
        logits, _ = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        logits.gather(1, class_idx[:, None]).sum().backward()
        alpha = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / cam.amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)
        return cam.detach(), logits.detach()
