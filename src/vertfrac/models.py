from __future__ import annotations
import torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm


class HFResNetBackbone(nn.Module):
    """Hugging Face `microsoft/resnet-50` (ResNet-50 v1.5, ImageNet-1k) as a feature extractor.
    Input (B,3,H,W) -> pooled features (B,2048). The ImageNet classifier is NOT used."""
    def __init__(self, name="microsoft/resnet-50", pretrained=True):
        super().__init__()
        from transformers import ResNetConfig, ResNetModel
        if pretrained:
            self.net = ResNetModel.from_pretrained(name)
        else:                                   # random init (e.g. when loading your own best.pt); default config = ResNet-50
            try: cfg = ResNetConfig.from_pretrained(name)
            except Exception: cfg = ResNetConfig()
            self.net = ResNetModel(cfg)
        self.num_features = self.net.config.hidden_sizes[-1]      # 2048

    def forward(self, x):
        return self.net(pixel_values=x).pooler_output.flatten(1)  # (B,2048,1,1) -> (B,2048)


class FractureNet(nn.Module):
    """Backbone -> {classification head (4 Genant grades), projection head (SupCon embedding)}."""
    def __init__(self, backbone="resnet18", pretrained=True, embed_dim=128, num_classes=4):
        super().__init__()
        w = "DEFAULT" if pretrained else None
        if backbone.startswith("hf:"):
            net = HFResNetBackbone(backbone[3:], pretrained)
            feat = net.num_features
            self._cam = "backbone.net.encoder.stages.3"           # last ResNet stage (7x7-ish map for 224 input)
        elif backbone.startswith("hf-hub:") or backbone.startswith("timm:"):
            # Hugging Face timm checkpoints, e.g. "hf-hub:timm/resnet50.a1_in1k" or
            # "hf-hub:timm/efficientnet_b0.ra_in1k". num_classes=0 -> pooled feature vector.
            import timm
            name = backbone.replace("timm:", "hf-hub:timm/") if backbone.startswith("timm:") else backbone
            net = timm.create_model(name, pretrained=pretrained, num_classes=0)
            feat = net.num_features
            self._cam = "backbone.layer4" if "resnet" in name else "backbone.conv_head"
        elif backbone.startswith("resnet"):
            net = getattr(tvm, backbone)(weights=w)
            feat = net.fc.in_features; net.fc = nn.Identity()
            self._cam = "backbone.layer4"
        elif backbone.startswith("efficientnet"):
            net = getattr(tvm, backbone)(weights=w)
            feat = net.classifier[1].in_features; net.classifier = nn.Identity()
            self._cam = "backbone.features"
        else:
            raise ValueError(backbone)
        self.backbone = net
        self.cls_head = nn.Linear(feat, num_classes)
        self.proj_head = nn.Sequential(nn.Linear(feat, feat), nn.ReLU(inplace=True), nn.Linear(feat, embed_dim))

    def forward(self, x):
        h = self.backbone(x)
        return self.cls_head(h), F.normalize(self.proj_head(h), dim=1)

    def gradcam_layer(self):
        m = self
        for p in self._cam.split("."):
            m = getattr(m, p)
        return m
