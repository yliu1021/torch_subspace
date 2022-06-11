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

def _convert_pruned_sv_to_bias(
    model: nn.Module,
    train_data,
    device
):
    def accumulate_pruned_output_hook(module: nn.Module, input, output):
        summed_output = torch.sum(output, dim=1)
        if not hasattr(module, 'pruned_output'):
            setattr(module, 'pruned_output', summed_output)
        else:
            module.pruned_output += summed_output
        return 
    
    def invert_mask_and_hook(_model, hook = None, handles = None):
        assert hook or handles != None, "At least 1 of hook or handles must be not None."
        if handles == None:
            handles = {}
        for name, module in _model.named_modules():
            if isinstance(module, SubspaceLR):
                if module.is_leaf:
                    module.sv_mask = 1 - module.sv_mask
                if hook != None:
                    handles[name] = module.register_forward_hook(hook)
                else:
                    handles[name].remove()
        return handles
    
    handles = invert_mask_and_hook(model, hook=accumulate_pruned_output_hook)

    # forward pass through data
    num_batches = len(train_data)
    model.eval()
    with torch.no_grad():
        for X, y in train_data:
            X, y = X.to(device), y.to(device)
            _ = model(X)

    for module in model.modules():
        if isinstance(module, SubspaceLR) and module.is_leaf:
            pruned_sv_bias = torch.mean(module.accumulated_pruned_output / num_batches)
            module.bias += pruned_sv_bias

    # Remove hook and invert mask again to get original mask
    invert_mask_and_hook(model, handles=handles)