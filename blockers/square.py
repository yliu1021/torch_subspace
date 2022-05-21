"""
Given a constant block size of `n * m = C`, we want to find `n` and `m`
such that we maximize the number of ranks we can keep maintaining sparsity
above some target.

The optimal solution is when `n = m = sqrt(C)`, hence, we do square blocking
where each square is as large as possible (bc the sparsity is `r / N` where `r`
is rank and `N` is block size).
"""
from torch import nn

from torch_subspace.lr import SubspaceLR


def make_blocks(model: nn.Module):
    # we call list because we're modifying the model's modules in place
    # (by adding more modules) so we don't want to enter an endless loop
    for module in list(model.modules()):
        if not isinstance(module, SubspaceLR):
            continue
        rows, cols = module.shape
        if cols >= rows:
            direction = "horizontal"
        else:
            direction = "vertical"
        small = min(rows, cols)
        large = max(rows, cols)
        block_sizes = [small] * (large // small)
        block_sizes[-1] += large % small
        module.split(block_sizes, direction)
