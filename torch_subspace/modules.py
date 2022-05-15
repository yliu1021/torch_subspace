from typing import Union

import math
import torch
from torch import nn
from torch.nn import functional as F

from .lr import SubspaceLR


class LinearLR(SubspaceLR):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype=torch.float,
        device=None,
    ) -> None:
        super().__init__(
            num_rows=out_features, num_cols=in_features, dtype=dtype, device=device
        )
        if bias:
            bias = torch.empty(out_features, dtype=dtype, device=device)
            bound = 1 / math.sqrt(self.num_cols)
            nn.init.uniform_(bias, -bound, bound)
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.eff_weights(), self.bias)


_2_tuple_or_int = Union[tuple[int, int], int]


def _make_pair(x):
    if not isinstance(x, int):
        return x
    return (x, x)


class Conv2dLR(SubspaceLR):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _2_tuple_or_int,
        stride: _2_tuple_or_int = 1,
        padding: Union[str, _2_tuple_or_int] = 0,
        dilation: _2_tuple_or_int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeroes",  # only support "zeroes" padding mode for now
        dtype=torch.float,
        device=None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _make_pair(kernel_size)
        self.stride = _make_pair(stride)
        self.padding = _make_pair(padding)
        self.dilation = _make_pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        super().__init__(
            num_rows=self.out_channels,
            num_cols=(self.in_channels * self.kernel_size[0] * self.kernel_size[1]),
            dtype=dtype,
            device=device,
        )
        if bias:
            bias = torch.empty(out_channels, dtype=dtype, device=device)
            bound = 1 / math.sqrt(self.num_cols)
            nn.init.uniform_(bias, -bound, bound)
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_shape = (
            self.out_channels,
            self.in_channels // self.groups,
            *self.kernel_size,
        )
        eff_weights = torch.reshape(self.eff_weights(), weight_shape)
        return F.conv2d(
            input=x,
            weight=eff_weights,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
