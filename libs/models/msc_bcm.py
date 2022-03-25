#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class MSC_BCM(nn.Module):
    """
    Multi-scale inputs with Box-driven Class-wise Masking(BCM).
    """

    def __init__(self, base, scales=None, n_classes=21):
        super(MSC_BCM, self).__init__()
        self.base = base
        if scales:
            self.scales = scales
        else:
            self.scales = [0.5, 0.75]

    def forward(self, x):
        # Original
        logits, bcm = self.base(x)
        bcm = F.sigmoid(bcm)# Normalize the range to 0-1.
        _, _, H, W = logits.shape
        interp = lambda l: F.interpolate(
            l, size=(H, W), mode="bilinear", align_corners=False
        )

        # Scaled
        logits_pyramid = []
        for p in self.scales:
            h = F.interpolate(x, scale_factor=p, mode="bilinear", align_corners=False)
            logits_pyra, _ = self.base(h)
            logits_pyramid.append(logits_pyra)

        # Pixel-wise max
        logits_all = [logits] + [interp(l) for l in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]
        logits_max = logits_max*bcm # Pixel-wise Masking.
        if self.training:
            return [logits] + logits_pyramid + [logits_max], bcm
        else:
            return logits_max, bcm
