"""
ALDS blocking: https://arxiv.org/pdf/2107.11442.pdf
Channels are grouped into `k` partitions.
"""
from torch_subspace.lr import SubspaceLR
from torch_subspace.modules import Conv2dLR


def make_blocks(module: SubspaceLR, k: int):
    if not isinstance(module, Conv2dLR):
        # ALDS is only for convolutions
        return
    unit_size = module.kernel_size[0] * module.kernel_size[1]
    block_sizes = [(module.in_channels + k - 1) // k * unit_size] * k
    block_sizes[-1] = (module.in_channels % k) * unit_size
    module.split(block_sizes, "horizontal")
