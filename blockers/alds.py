"""
ALDS blocking: https://arxiv.org/pdf/2107.11442.pdf
Channels are grouped into `k` partitions.
"""
from torch import nn

from torch_subspace.lr import SubspaceLR
from torch_subspace.modules import Conv2dLR, LinearLR


def make_blocks(model: nn.Module, k: int):
    # we call list because we're modifying the model's modules in place
    # (by adding more modules) so we don't want to enter an endless loop
    for module in list(model.modules()):
        if not isinstance(module, SubspaceLR):
            continue
        if isinstance(module, Conv2dLR):
            # ALDS is only for convolutions
            unit_size = module.kernel_size[0] * module.kernel_size[1]
            block_sizes = [(module.in_channels + k - 1) // k * unit_size] * k
            block_sizes[-1] = (module.in_channels % k) * unit_size
        elif isinstance(module, LinearLR):
            # but we extend it to linear layers for a fair comparison
            block_sizes = [(module.num_cols + k - 1) // k] * k
            block_sizes[-1] = module.num_cols % k
        else:
            raise ValueError("Got unsupported SubspaceLR")
        module.split(block_sizes, "horizontal")
