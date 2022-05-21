"""
Prunes each block module based on relative error:
(singular value)_i / max (singular value)
"""
import numpy as np
import torch
from torch import nn

from torch_subspace import SubspaceLR

from ._helper import _prune_scores


def singular_values(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(x).detach()


def _compute_scores(model: nn) -> list[np.ndarray]:
    prunable_modules = [
        m for m in model.modules() if isinstance(m, SubspaceLR) and m.is_leaf
    ]
    scores = []
    for i, module in enumerate(prunable_modules):
        print(f"\rScoring module: {i+1} / {len(prunable_modules)}", end="")
        svs = singular_values(module.eff_weights())
        svs /= svs.max()
        scores.append(np.array(svs.cpu()))
    print()
    return scores


def prune(
    model: nn.Module, sparsity: float, device=None, *args, **kwargs
) -> list[np.ndarray]:
    with torch.no_grad():
        scores = _compute_scores(model)
    _prune_scores(model, scores, sparsity, device)
    return scores
