"""Neural networks: a 2D U-Net segmenter and a ResNet-18 classifier.

Both contain dropout layers that stay active at inference time when
:func:`enable_mc_dropout` is called, which gives Monte-Carlo dropout
uncertainty estimates (several stochastic passes, then mean and spread).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Classic 4-level U-Net. Outputs per-pixel logits for ``num_classes`` labels."""

    def __init__(self, in_channels: int, num_classes: int, base: int = 32, dropout: float = 0.2):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8]
        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in chs:
            self.encoders.append(ConvBlock(prev, ch))
            prev = ch
        self.bottleneck = ConvBlock(chs[-1], chs[-1] * 2, dropout=dropout)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = chs[-1] * 2
        for ch in reversed(chs):
            self.ups.append(nn.ConvTranspose2d(prev, ch, 2, stride=2))
            self.decoders.append(ConvBlock(ch * 2, ch, dropout=dropout / 2))
            prev = ch
        self.head = nn.Conv2d(chs[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


class Classifier(nn.Module):
    """ResNet-18 adapted to grayscale input, with dropout before the final layer."""

    def __init__(self, num_classes: int, in_channels: int = 1, dropout: float = 0.3, pretrained: bool = False):
        super().__init__()
        weights = None
        if pretrained:
            from torchvision.models import ResNet18_Weights

            weights = ResNet18_Weights.IMAGENET1K_V1
        net = resnet18(weights=weights)
        if in_channels != 3:
            old = net.conv1
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    net.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, num_classes))
        self.net = net

    @property
    def cam_layer(self) -> nn.Module:
        return self.net.layer4

    def forward(self, x):
        return self.net(x)


def enable_mc_dropout(model: nn.Module) -> None:
    """Eval mode everywhere (frozen batch-norm stats) except dropout layers."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()


def grad_cam(model: Classifier, x: torch.Tensor, target: int) -> torch.Tensor:
    """Grad-CAM heatmap for class ``target``, shape (H, W) in [0, 1], matching ``x``."""
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = model.cam_layer
    h1 = layer.register_forward_hook(lambda m, i, o: activations.append(o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = model(x)
            logits[0, target].backward()
    finally:
        h1.remove()
        h2.remove()
    act, grad = activations[0].detach(), gradients[0].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return cam / cam.max() if cam.max() > 0 else cam


class ConvBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """3D U-Net (nnU-Net style: instance norm, leaky ReLU) for whole-volume segmentation.

    Input sizes must be divisible by 16 (four poolings); use patches of 96 or 128.
    """

    def __init__(self, in_channels: int, num_classes: int, base: int = 16, dropout: float = 0.1):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8]
        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in chs:
            self.encoders.append(ConvBlock3d(prev, ch))
            prev = ch
        self.bottleneck = ConvBlock3d(chs[-1], chs[-1] * 2, dropout=dropout)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = chs[-1] * 2
        for ch in reversed(chs):
            self.ups.append(nn.ConvTranspose3d(prev, ch, 2, stride=2))
            self.decoders.append(ConvBlock3d(ch * 2, ch))
            prev = ch
        self.head = nn.Conv3d(chs[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = F.max_pool3d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


class NoduleClassifier3D(nn.Module):
    """Small 3D CNN that scores a 48 mm cube around a lung nodule (benign vs malignant)."""

    def __init__(self, num_classes: int = 2, base: int = 16, dropout: float = 0.3):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8]
        layers, prev = [], 1
        for ch in chs:
            layers += [ConvBlock3d(prev, ch), nn.MaxPool3d(2)]
            prev = ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(prev, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def sliding_window_3d(model: nn.Module, volume: torch.Tensor, patch: int = 128, overlap: float = 0.5, flip_tta: bool = True) -> torch.Tensor:
    """Softmax probabilities for a (C, Z, Y, X) volume using overlapping patches.

    Patches are blended with a Gaussian weight so seams don't show; with
    ``flip_tta`` each patch is also predicted left-right mirrored and averaged.
    """
    c, *shape = volume.shape
    pad = [max(0, patch - s) for s in shape]
    vol = F.pad(volume, (0, pad[2], 0, pad[1], 0, pad[0]))
    shape_p = vol.shape[1:]
    step = max(1, int(patch * (1 - overlap)))
    starts = [sorted({*range(0, max(s - patch, 0) + 1, step), max(s - patch, 0)}) for s in shape_p]
    g = torch.exp(-((torch.arange(patch, dtype=torch.float32) - (patch - 1) / 2) ** 2) / (2 * (patch / 8) ** 2))
    weight = (g[:, None, None] * g[None, :, None] * g[None, None, :]).to(volume.device).clamp_min(1e-3)
    probs = None
    norm = torch.zeros(shape_p, device=volume.device)
    model.eval()
    # Half precision on NVIDIA GPUs: about twice as fast, same outlines in practice.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=volume.is_cuda):
        for z in starts[0]:
            for y in starts[1]:
                for x in starts[2]:
                    tile = vol[None, :, z : z + patch, y : y + patch, x : x + patch]
                    p = torch.softmax(model(tile).float(), dim=1)
                    if flip_tta:
                        p = (p + torch.softmax(model(tile.flip(-1)).float(), dim=1).flip(-1)) / 2
                    if probs is None:
                        probs = torch.zeros((p.shape[1], *shape_p), device=volume.device)
                    probs[:, z : z + patch, y : y + patch, x : x + patch] += p[0] * weight
                    norm[z : z + patch, y : y + patch, x : x + patch] += weight
    probs = probs / norm.clamp_min(1e-6)
    return probs[:, : shape[0], : shape[1], : shape[2]]
