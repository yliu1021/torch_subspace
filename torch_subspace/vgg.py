from typing import Any, Dict, List, Union, cast

import torch
import torch.nn as nn
from torchvision.models import vgg

from torch_subspace.lr import SubspaceLR

from .modules import Conv2dLR, LinearLR


class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            LinearLR(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            LinearLR(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            LinearLR(4096, num_classes),
        )
        for m in self.modules():
            # vgg uses different initialization than the "standard"
            if isinstance(m, LinearLR):
                nn.init.normal_(m._weights[0][0][0], 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Conv2dLR):
                fan_out = m.out_channels * m.kernel_size[0] * m.kernel_size[1]
                gain = 2**0.5  # ReLU gain = sqrt(2)
                std = gain / (fan_out**0.5)
                nn.init.normal_(m._weights[0][0][0], 0, std)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = Conv2dLR(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# fmt: off
cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
# fmt: on


def _vgg(
    cfg: str, batch_norm: bool, num_classes: int, dropout: float, device=None
) -> VGG:
    model = VGG(
        features=make_layers(cfgs[cfg], batch_norm=batch_norm),
        num_classes=num_classes,
        dropout=dropout,
    )
    model = model.to(device)
    return model


def vgg11(batch_norm: bool, num_classes: int, dropout: float = 0.5, device=None) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    """
    return _vgg(
        "A", batch_norm, num_classes=num_classes, dropout=dropout, device=device
    )


def vgg13(batch_norm: bool, num_classes: int, dropout: float = 0.5, device=None) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    """
    return _vgg(
        "B", batch_norm, num_classes=num_classes, dropout=dropout, device=device
    )


def vgg16(batch_norm: bool, num_classes: int, dropout: float = 0.5, device=None) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    """
    return _vgg(
        "D", batch_norm, num_classes=num_classes, dropout=dropout, device=device
    )


def vgg19(batch_norm: bool, num_classes: int, dropout: float = 0.5, device=None) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    """
    return _vgg(
        "E", batch_norm, num_classes=num_classes, dropout=dropout, device=device
    )
