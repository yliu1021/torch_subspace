"""
Prunes each block module based on relative layer error:
(singular value)_i / max (singular value of layer's effective weights)
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
    roots = {}
    for module in prunable_modules:
        root = module
        while root.parent_module is not None:
            root = root.parent_module[0]
        roots[module] = root
    max_svs = {}
    for root in roots.values():
        if root is max_svs:
            continue
        svs = torch.linalg.svdvals(root.eff_weights()).detach()
        max_svs[root] = svs[0]
    scores = []
    for i, module in enumerate(prunable_modules):
        print(f"\rScoring module: {i+1} / {len(prunable_modules)}", end="")
        svs = singular_values(module.eff_weights())
        svs /= max_svs[roots[module]]
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
