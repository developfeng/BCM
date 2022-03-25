from __future__ import absolute_import
from .resnet import *
from .deeplabv2 import *
from .msc_bcm import *


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def ResNet101(n_classes):
    return ResNet(n_classes=n_classes, n_blocks=[3, 4, 23, 3])

def DeepLabV2_ResNet101_MSC_BCM(n_classes):
    return MSC_BCM(
        base=DeepLabV2(
            n_classes=n_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
        ),
        scales=[0.5, 0.75], 
        n_classes = 21,
    )