import numpy as np
import torch
from torch import nn

from torch_subspace import SubspaceLR


def _prune_scores(
    model: nn.Module, scores: list[np.ndarray], sparsity: float, device=None
):
    all_scores = np.concatenate(scores)
    all_scores = np.sort(all_scores)
    threshold = all_scores[int(sparsity * len(all_scores))]
    # print some basic stats about the scores
    print(f"Got {len(all_scores)} scores")
    print("Score min / max / mean / std:")
    print(
        f"{all_scores.min():.4f} {all_scores.max():.4f} {all_scores.mean():.4f} {all_scores.std():.4f}"
    )
    percentiles = [0.25, 0.5, 0.75, 0.9]
    percentile_values = [all_scores[int(p * len(all_scores))] for p in percentiles]
    print(f"Values at percentiles {percentiles}:")
    print(percentile_values)
    print(f"Setting threshold to: {threshold}")
    prunable_modules = [
        module
        for module in model.modules()
        if isinstance(module, SubspaceLR) and module.is_leaf
    ]
    assert len(prunable_modules) == len(scores)
    for i, (module, mask_scores) in enumerate(zip(prunable_modules, scores)):
        mask = torch.tensor(mask_scores >= threshold)
        module.set_mask(mask.to(device=device))
        print(f"\rPruning module {i+1} / {len(prunable_modules)}", end="")
    print()
