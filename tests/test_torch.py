import pytest
torch = pytest.importorskip("torch")


def test_supcon_and_model_forward():
    from vertfrac.losses import SupConLoss, CombinedLoss, make_ce
    from vertfrac.models import FractureNet
    from vertfrac.gradcam import GradCAM
    z = torch.nn.functional.normalize(torch.randn(16, 8), dim=1); y = torch.randint(0, 4, (16,))
    assert torch.isfinite(SupConLoss(0.1)(z, y))
    net = FractureNet("resnet18", pretrained=False)
    x = torch.randn(4, 3, 96, 96); lo, e = net(x)
    assert lo.shape == (4, 4) and e.shape == (4, 128)
    loss, _ = CombinedLoss(make_ce([50, 10, 5, 2]), 0.5)(lo, e, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    cam, _ = GradCAM(net, net.gradcam_layer())(x)
    assert cam.shape == (4, 1, 96, 96) and cam.min() >= 0 and cam.max() <= 1


def test_hf_resnet_backbone_and_gradcam():
    pytest.importorskip("transformers")
    from vertfrac.models import FractureNet
    from vertfrac.gradcam import GradCAM
    net = FractureNet("hf:microsoft/resnet-50", pretrained=False)
    x = torch.randn(2, 3, 96, 96); lo, e = net(x)
    assert lo.shape == (2, 4) and e.shape == (2, 128)
    cam, _ = GradCAM(net, net.gradcam_layer())(x)
    assert cam.shape == (2, 1, 96, 96)
