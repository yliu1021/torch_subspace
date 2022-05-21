"""
Alignment pruning scores each singular vector based on how much it perturbs
the output
"""
import numpy as np
import torch
from torch import nn
from torch.utils import data

from torch_subspace import SubspaceLR

from ._helper import _prune_scores


def _compute_scores(
    model: nn.Module, train_data: data.DataLoader, device=None
) -> list[np.ndarray]:
    """Prunes the model in place"""
    sample_in, _ = next(iter(train_data))
    sample_in = sample_in[:16].detach()
    sample_in = sample_in.to(device)
    baseline_output = model(sample_in)

    def eval_score() -> float:
        new_output = model(sample_in)
        score = torch.mean((baseline_output - new_output) ** 2).cpu()
        return score.item()

    def compute_block_scores(module: SubspaceLR, module_ind: int) -> np.ndarray:
        if module.sv_mask is None:
            mask = torch.ones(module.max_rank())
        else:
            mask = module.sv_mask.clone()
        scores = []
        for i, mask_val in list(enumerate(mask)):
            print(f"\rScoring module {module_ind+1} / {len(prunable_modules)} (mask {i:>4} / {len(mask)})", end="")
            if mask_val == 0:  # don't touch masks that are already set
                scores.append(0)
            else:
                mask[i] = 0
                module.set_mask(mask.to(device=device))
                scores.append(eval_score())
                mask[i] = 1
                module.set_mask(mask.to(device=device))
        return np.array(scores)

    scores = []
    prunable_modules = [
        module
        for module in model.modules()
        if isinstance(module, SubspaceLR) and module.is_leaf
    ]
    for i, module in enumerate(prunable_modules):
        scores.append(compute_block_scores(module, i))
    print()
    return scores


def prune(
    model: nn.Module, train_data: data.DataLoader, sparsity: float, device=None
) -> list[np.ndarray]:
    with torch.no_grad():
        scores = _compute_scores(model, train_data, device)
    _prune_scores(model, scores, sparsity, device)
    return scores
