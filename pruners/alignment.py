"""
- We want the max compression to reach at least `(1 - rho)` (so keep at least rho proportion of the weights)
- We want to achieve this with `R` minimum rank per block
- If we assume each block is approximately square with size `N x N``
- Compression ratio of each block at rank R is `R * (N + N) / (N * N) = 2 R / N`
- This means we need `2 R / N < rho` which means `2 R / rho < N`
- This means block size has to be at least `2 R / rho` big
"""
import numpy as np

import torch
from torch import nn
from torch.utils import data

from torch_subspace import SubspaceLR


R = 1
rho = 1 - 0.95  # wish to prune to at least 90% sparsity
min_block_size = int(2 * R / rho)


def _partition(min_size: int, tot_size: int) -> list[int]:
    assert min_size <= tot_size
    res = [min_size] * (tot_size // min_size)
    res[-1] += tot_size % min_size
    return res


def _make_blocks(subspace_layer: SubspaceLR) -> tuple[list[int], list[int]]:
    n, m = subspace_layer.shape
    r = min(n, m)
    # M_max = max(n, m)
    # if r < min_block_size:
    #     """
    #     If one of the dimensions is less than `min_block_size`, then we need to
    #     recompute the size along the other dimension to ensure compression ratio.
    #     If the size along the other dimension is M, then compression will be
    #     `R * (r + M) / (r * M) = R / M + R / r < rho`
    #     which implies
    #     `R / (rho - R / r) < M`
    #     so
    #     `r R / (r * rho - R) < M`
    #     """
    #     if R / M_max + R / r >= rho:
    #         # can't satisfy the required sparsity but we'll get as close as possible
    #         M = M_max
    #     else:
    #         M = int(r * R / (r * rho - R))
    #     if n < m:
    #         block_size = (r, M)
    #     else:
    #         block_size = (M, r)
    # else:
    #     block_size = (min_block_size, min_block_size)
    block_size = (r, r)
    return (_partition(block_size[0], n), _partition(block_size[1], m))


def make_blocks(model: nn.Module):
    with torch.no_grad():
        for i, module in list(enumerate(model.modules())):
            if not isinstance(module, SubspaceLR):
                continue
            row_sizes, col_sizes = _make_blocks(module)
            module.set_blocks(row_sizes, col_sizes)


def _prune(model: nn.Module, train_data: data.DataLoader, device=None):
    """Prunes the model in place"""
    sample_in, _ = next(iter(train_data))
    sample_in = sample_in.to(device)
    baseline_output = model(sample_in)

    def eval_score():
        new_output = model(sample_in)
        score = torch.mean((baseline_output - new_output) ** 2).cpu()
        return score.item()

    def compute_block_scores(module: SubspaceLR, block_row: int, block_col: int):
        mask = module.create_set_mask(block_row, block_col)
        scores = []
        for i, mask_val in list(enumerate(mask)):
            if mask_val == 0:  # don't touch masks that are already set
                scores.append(0)
            else:
                mask[i] = 0
                module.set_mask(block_row, block_col, mask.to(device=device))
                scores.append(eval_score())
                mask[i] = 1
                module.set_mask(block_row, block_col, mask.to(device=device))
        return np.array(scores)

    scores = []
    for module in model.modules():
        if not isinstance(module, SubspaceLR):
            continue
        print(f"Evaluating module: {len(scores) + 1}")
        module_scores = []
        for row in range(module.block_shape[0]):
            module_row_scores = []
            for col in range(module.block_shape[1]):
                print(f"\rEvaluating block {row} x {col}", end="")
                block_scores = compute_block_scores(module, row, col)
                module_row_scores.append(block_scores)
            module_scores.append(module_row_scores)
        print()
        scores.append(module_scores)
    return scores


def prune(model: nn.Module, train_data: data.DataLoader, device=None):
    with torch.no_grad():
        scores = _prune(model, train_data, device)
    return scores
