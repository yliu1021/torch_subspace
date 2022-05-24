"""
We sample different masks and accumulate (per mask) the average error term
when that mask is turned off. The average error per mask is the score for that
mask.
"""
import numpy as np
import torch
from torch import nn
from torch.utils import data

from torch_subspace import SubspaceLR

from ._helper import _prune_scores


def _compute_scores(
    model: nn.Module, train_data: data.DataLoader, sparsity: float, device=None
) -> list[np.ndarray]:
    """Prunes the model in place"""
    sample_in, _ = next(iter(train_data))
    sample_in = sample_in.to(device)
    baseline_output = model(sample_in)

    def eval_score() -> float:
        new_output = model(sample_in)
        score = torch.mean((baseline_output - new_output) ** 2).cpu()
        return score.item()

    prunable_modules = [
        module
        for module in model.modules()
        if isinstance(module, SubspaceLR) and module.is_leaf
    ]
    singular_values = [
        torch.linalg.svdvals(module.eff_weights()).cpu().detach().numpy()
        for module in prunable_modules
    ]

    def compute_probabilities(p: float) -> list[np.ndarray]:
        # normalize and apply p norm
        probabilities = [(prob / prob.max()) ** p for prob in singular_values]
        # scale everything down to probability measure
        tot_energy = sum(np.sum(prob) for prob in probabilities)
        num_elements = sum(len(prob) for prob in probabilities)
        scale = ((1 - sparsity) * num_elements / tot_energy)
        return [np.clip(prob * scale, 0, 1) for prob in probabilities]

    num_samples = [np.zeros(module.max_rank()) for module in prunable_modules]
    raw_scores = [np.zeros(module.max_rank()) for module in prunable_modules]
    for i in range(1000):  # perform 1000 samples (TODO: decide how many iterations)
        print(f"\rIter: {i}", end="")
        probabilities = compute_probabilities(p=2)  # TODO: pick better p value
        masks = [np.zeros(len(prob)) for prob in probabilities]
        while any(m.sum() < 1 for m in masks):
            # loop until every mask has at least one singular value turned on
            masks = [np.random.binomial(n=1, p=prob) for prob in probabilities]
        # accumulate how many times we turned off each mask
        num_samples = [s + (1 - m) for s, m in zip(num_samples, masks)]
        for module, mask in zip(prunable_modules, masks):
            module.set_mask(torch.tensor(mask, device=device))
        score = eval_score()
        # accumulate the score for each mask we turned off
        raw_scores = [r + score * (1 - m) for r, m in zip(raw_scores, masks)]
    print()
    scores = [r / n for r, n in zip(raw_scores, num_samples)]
    return scores


def prune(
    model: nn.Module,
    train_data: data.DataLoader,
    sparsity: float,
    device=None,
    *args,
    **kwargs,
) -> list[np.ndarray]:
    with torch.no_grad():
        scores = _compute_scores(model, train_data, sparsity, device)
    _prune_scores(model, scores, sparsity, device)
    return scores
