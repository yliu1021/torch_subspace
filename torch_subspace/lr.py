"""
SubpaceLR represents an abstract nn.Module to subclass by other classes in order
to inherit subspace low rank decomposable properties. SubspaceLR defines a 
"default" forward pass that just performs a matrix multiplication.

SubspaceLR has two representations:
1. A single matrix forward pass (expressed in W or UV form)
2. A recursive representation of a partitioned subspace

In representation 1, the `weights` attribute is a nn.ParameterList
with either one or two weights (corresponding to W or UV form), and an
attribute `sv_mask` for masking.

In representation 2, the `weights` attribute is a nn.ModuleList containing
(recursively) a list of SubspaceLR modules and a `direction` attribute
that represents the axis along which these SubspaceLR modules span in the
effective matrix. This allows each child SubspaceLR to represent a block
of the parent effective matrix.

* Note that direction is either "horizontal" or "vertical"

In representation 1, we call the SubspaceLR module a "leaf" module, and in
representation 2, we call it a "non-leaf" module. Setting (or clearing) a
mask is only supported on "leaf" modules.

To go between the two representations, the methods `split` and `collect`
will go from representation 1 to 2 and from 2 to 1 respectively.
"""
from typing import Union

import torch
from torch import nn
from torch.nn import functional as F


def _decompose(parameter: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Performs SVD and returns (U, V) with sqrt(s) multiplied in"""
    u, s, v = torch.linalg.svd(parameter, full_matrices=False)
    s_sqrt = torch.diag(torch.sqrt(s))
    # make sure we return a "fresh" set of U and V matrices
    u = torch.matmul(u, s_sqrt).detach()
    v = torch.matmul(s_sqrt, v).detach()
    return u, v


def _leaf_only(fn):
    """Helper decorator to make a method callable only on a leaf node"""

    def f(self: "SubspaceLR", *args, **kwargs):
        if not self.is_leaf:
            raise ValueError(f"method: {fn.__name__} can only be called on a leaf node")
        return fn(self, *args, **kwargs)

    return f


class SubspaceLR(nn.Module):
    def __init__(self, num_rows: int, num_cols: int):
        """
        Creates a subspace linear module with a given num_rows and num_cols for the
        effective weight matrix size. For Linear modules, this corresponds to the
        number of outputs and number of inputs. For Conv modules, reshaping is required
        on the eff_weight matrix.
        """
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._direction = "horizontal"  # horizontal split by default
        self.weights: Union[nn.ParameterList, nn.ModuleList] = nn.ParameterList(
            [nn.Parameter(torch.empty(self.num_rows, self.num_cols))]
        )
        self.register_buffer("sv_mask", None)

    @property
    def is_leaf(self) -> bool:
        """
        A leaf node can be decomposed from W form to UV form. Non-leaf nodes contains
        blocked low rank matrices (recursively)
        """
        if isinstance(self.weights, nn.ParameterList):
            return True
        elif isinstance(self.weights, nn.ModuleList):
            return False
        else:
            raise RuntimeError("self.weights must be a parameter list or module list")

    @_leaf_only
    def set_mask(self, new_mask: torch.Tensor):
        """
        Sets the singular vector mask to a 1D vector. Automatically converts the layer
        into UV mode.
        """
        assert new_mask.shape == (
            self.max_rank(),
        ), f"self.set_mask(...) must take a mask of shape ({self.max_rank()},)"
        self.sv_mask = new_mask.detach().clone()
        if len(self.weights) == 1:
            u, v = _decompose(self.weights[0])
            u = nn.Parameter(u)
            v = nn.Parameter(v)
            self.weights = nn.ParameterList([u, v])
        assert (
            len(self.weights) == 2
        ), "self.weights should be a parameter list with two elements after setting mask"

    @_leaf_only
    def clear_mask(self):
        """
        Clears the mask and converts the layer into W mode
        """
        self.weights = nn.ParameterList([nn.Parameter(self.eff_weights().detach())])
        self.register_buffer("sv_mask", None)

    def eff_weights(self) -> torch.Tensor:
        """Returns the effective weights of the layer"""
        if self.is_leaf:
            if len(self.weights) == 1:
                return self.weights[0]
            elif len(self.weights) == 2:
                return self.weights[0] @ self.weights[1]
            else:
                raise RuntimeError("Can't have more than 2 weights")
        else:
            assert all(isinstance(subspace, SubspaceLR) for subspace in self.weights)
            eff_weights = [subspace.eff_weights() for subspace in self.weights]
            if self._direction == "horizontal":
                eff_weights = torch.concat(eff_weights, dim=1)
            elif self._direction == "vertical":
                eff_weights = torch.concat(eff_weights, dim=0)
            else:
                raise ValueError("self.direction must be 'horizontal' or 'vertical'")
            assert eff_weights.shape == (self.num_rows, self.num_cols)
            return eff_weights

    def set_eff_weights(self, eff_weights: torch.Tensor):
        """
        Sets the effective weights of the layer regardless if it's in W mode or
        UV mode.
        """
        assert eff_weights.shape == (self.num_rows, self.num_cols)
        eff_weights = eff_weights.detach().clone()
        if self.is_leaf:
            if len(self.weights) == 1:
                self.weights = nn.ParameterList([nn.Parameter(eff_weights)])
            elif len(self.weights) == 2:
                u, v = _decompose(eff_weights)
                u = nn.Parameter(u)
                v = nn.Parameter(v)
                self.weights = nn.ParameterList([u, v])
                self.sv_mask = torch.ones(self.max_rank())
            else:
                raise RuntimeError("self.weights must have at most two elements")
        else:
            if self._direction == "horizontal":
                ind = 0
                for subspace in self.weights:
                    assert isinstance(subspace, SubspaceLR)
                    subspace.set_eff_weights(
                        eff_weights[:, ind : ind + subspace.num_cols]
                    )
                    ind += subspace.num_cols
            elif self._direction == "vertical":
                ind = 0
                for subspace in self.weights:
                    assert isinstance(subspace, SubspaceLR)
                    subspace.set_eff_weights(
                        eff_weights[ind : ind + subspace.num_rows, :]
                    )
                    ind += subspace.num_rows
            else:
                raise ValueError("self.direction must be 'horizontal' or 'vertical'")

    def max_rank(self) -> int:
        """The maximum rank this layer can have"""
        return min(self.num_rows, self.num_cols)

    def split(self, block_sizes: list[int], direction: str):
        """Splits the layer into subspaces in `direction` whose sizes are specified by `block_sizes`"""
        if direction == "horizontal" and sum(block_sizes) != self.num_cols:
            raise ValueError(f"Block sizes must sum up to {self.num_cols}")
        if direction == "vertical" and sum(block_sizes) != self.num_rows:
            raise ValueError(f"Block sizes must sum up to {self.num_rows}")
        eff_weights = self.eff_weights().detach()
        if direction == "horizontal":
            self.weights = nn.ModuleList(
                [
                    SubspaceLR(num_rows=self.num_rows, num_cols=block_size)
                    for block_size in block_sizes
                ]
            )
        elif direction == "vertical":
            self.weights = nn.ModuleList(
                [
                    SubspaceLR(num_rows=block_size, num_cols=self.num_cols)
                    for block_size in block_sizes
                ]
            )
        else:
            raise ValueError("Direction must be 'horizontal' or 'vertical'")
        self._direction = direction
        self.register_buffer("sv_mask", None)
        self.set_eff_weights(eff_weights)

    def collect(self):
        """Collects all subspaces and puts them into W form. If we're already in UV form, this is the same as clear_mask"""
        if self.is_leaf:
            self.clear_mask()
        else:
            self.weights = nn.ParameterList([nn.Parameter(self.eff_weights().detach())])
            self.register_buffer("sv_mask", None)

    def numels(self) -> int:
        """The effective number of parameters in this layer accounting for masks"""
        if self.is_leaf:
            if len(self.weights) == 1:
                return self.weights[0].numel()
            elif len(self.weights) == 2:
                u, v = self.weights
                mask_sparsity = (self.sv_mask == 0).sum().item()
                return (u.numel() + v.numel()) * mask_sparsity // self.max_rank()
            else:
                raise RuntimeError("self.weights must have at most two elements")
        else:
            return sum(subspace.numels() for subspace in self.weights)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_rows, self.num_cols)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return F.linear(x, self.eff_weights())
