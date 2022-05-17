import numpy as np
import torch
from torch import nn

from torch_subspace import SubspaceLR


def prune_scores(
    model: nn.Module, scores: list[list[list[np.ndarray]]], sparsity: float, device=None
):
    all_scores = []
    for layer_scores in scores:
        for layer_row_scores in layer_scores:
            for mask_scores in layer_row_scores:
                all_scores.append(mask_scores)
    all_scores = np.concatenate(all_scores)
    all_scores = np.sort(all_scores)
    threshold = all_scores[int(sparsity * len(all_scores))]

    score_ind = 0
    for module in model.modules():
        if not isinstance(module, SubspaceLR):
            continue
        layer_scores = scores[score_ind]
        score_ind += 1

        for row in range(module.block_shape[0]):
            for col in range(module.block_shape[1]):
                mask_scores = layer_scores[row][col]
                mask = torch.tensor(mask_scores >= threshold)
                module.set_mask(row, col, mask.to(device=device))
