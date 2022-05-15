"""
SubpaceLR represents an abstract nn.Module to subclass by other classes in order
to inherit subspace low rank decomposable properties.

The number of blocks is mutable and the SVD mask for each block is mutable.
When a mask is set to None, that block is represented as a single matrix W.
When a mask is set to a 1D tensor, that block is represented as two matrices U/V.

** IMPORTANT **
When nn.Modules are subclassed, the class keeps track of modules, parameters, and
buffers used by the class in question. It does this by *tracking the Python object's
attributes* (see the `__setattr__` method of nn.Module).

Unfortunately, the issue with this is if our attribute is a list of lists, then
PyTorch will *not* automatically track parameters in that list. This means each
weight block and mask will need to be registered via `register_parameter(name, param)`
or `register_buffer(name, tensor)` (see nn.ParameterList and nn.ModuleList).

Thus, a two nested nn.ModuleLists are used to store a nn.ParameterList. Effectively,
each block is represented as a nn.ParameterList that contains one parameter if 
that block is full rank and two parameters if that block is low rank.
"""
import math
from typing import Optional

import torch
from torch import nn


def _decompose(parameter: nn.Parameter) -> tuple[nn.Parameter, nn.Parameter]:
    """Performs SVD and returns (U, V) with sqrt(s) multiplied in"""
    u, s, v = torch.linalg.svd(parameter, full_matrices=False)
    s_sqrt = torch.diag(torch.sqrt(s))
    # make sure we return a "fresh" set of U and V matrices
    u = torch.matmul(u, s_sqrt).detach()
    v = torch.matmul(s_sqrt, v).detach()
    return (nn.Parameter(u), nn.Parameter(v))


class SubspaceLR(nn.Module):
    def __init__(self, num_rows: int, num_cols: int, dtype=torch.float, device=None):
        """
        Creates a subspace linear module with a given num_rows and num_cols for the
        effective weight matrix size. For Linear modules, this corresponds to the
        number of outputs and number of inputs. For Conv modules, reshaping is required
        on the eff_weight matrix.
        """
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.dtype = dtype
        self.device = device
        # initialize weights
        init_weights = torch.empty(
            self.num_rows, self.num_cols, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(init_weights, a=math.sqrt(5))
        init_weights = nn.Parameter(init_weights)
        self._weights = nn.ModuleList(
            [nn.ModuleList([nn.ParameterList([init_weights])])]
        )
        self._masks: list[list[Optional[torch.Tensor]]] = [[None]]

    def set_blocks(
        self,
        row_sizes: list[int],
        col_sizes: list[int],
    ):
        """
        Breaks down the effective weight matrix into blocks, suitable for
        subspace low rank decomposition
        """
        if sum(row_sizes) != self.num_rows or sum(col_sizes) != self.num_cols:
            raise ValueError(
                f"The row and col sizes must sum up to self.num_rows and self.num_cols respectively"
            )
        eff_weights = self.eff_weights()
        self._weights = nn.ModuleList()
        self._masks = []
        row = 0
        for row_size in row_sizes:
            weight_row = nn.ModuleList()
            mask_row = []
            col = 0
            for col_size in col_sizes:
                weight_row.append(
                    nn.ParameterList(
                        [
                            nn.Parameter(
                                eff_weights[row : row + row_size, col : col + col_size]
                            )
                        ]
                    )
                )
                mask_row.append(None)
                col += col_size
            self._weights.append(weight_row)
            self._masks.append(mask_row)
            row += row_size

    def set_mask(self, row_ind: int, col_ind: int, mask: torch.Tensor):
        """Sets the mask for a given block in the weight matrix"""
        if len(mask.shape) != 1:
            raise ValueError("Mask must be 1 dimensional")
        mask = mask.to(dtype=self.dtype, device=self.device)
        block = self._weights[row_ind][col_ind]
        if self._masks[row_ind][col_ind] is None:
            assert (
                len(block) == 1
            ), "Mask is none for this block so block should be a single weight matrix"
            block = block[0]
            if mask.shape[0] != min(block.shape):
                raise ValueError("Mask length must equal min(block.shape)")
            u, v = _decompose(block)
            self._weights[row_ind][col_ind] = nn.ParameterList([u, v])
            self._masks[row_ind][col_ind] = mask.detach()
        else:
            assert (
                len(block) == 2
            ), "Mask is not none for this block so block should be two matrices"
            if mask.shape[0] != self._masks[row_ind][col_ind].shape[0]:
                raise ValueError("Mask length must equal min(block.shape)")
            self._masks[row_ind][col_ind] = mask.detach()

    def clear_mask(self, row_ind: int, col_ind: int):
        """Removes the mask of a block. No op if mask was already cleared"""
        if self._masks[row_ind][col_ind] is None:
            return
        [u, v] = self._weights[row_ind][col_ind]
        old_mask = self._masks[row_ind][col_ind]
        old_eff_block = u @ torch.diag(old_mask) @ v
        self._weights[row_ind][col_ind] = nn.ParameterList(
            [nn.Parameter(old_eff_block.detach())]
        )
        self._masks[row_ind][col_ind] = None

    def eff_weights(self) -> torch.Tensor:
        """
        Returns the effective weights of the layer (with grads)
        """
        weights = []
        for row_ind, block_row in enumerate(self._weights):
            weight_row = []
            for col_ind, block in enumerate(block_row):
                block_mask = self._masks[row_ind][col_ind]
                if block_mask is None:
                    assert (
                        len(block) == 1
                    ), "The block has no mask so it should have one weight matrix"
                    weight_row.append(block[0])
                else:
                    assert (
                        len(block) == 2
                    ), "The block has a mask so it should have two weight matrices"
                    [u, v] = block
                    weight_row.append(u @ torch.diag(block_mask) @ v)
            weights.append(torch.concat(weight_row, dim=1))
        return torch.concat(weights, dim=0)

    @property
    def block_shape(self) -> tuple[int, int]:
        """Returns the number of blocks in (num_vertical, num_horizontal)"""
        num_rows = len(self._weights)
        num_cols = len(self._weights[0])
        return (num_rows, num_cols)

    def block_size(self, row: int, col: int) -> tuple[int, int]:
        """Returns the size of the block at a given index"""
        block = self._weights[row][col]
        if len(block) == 1:
            return block[0].shape
        else:
            [u, v] = block
            return (u.shape[0], v.shape[1])

    def size(self) -> tuple[int, int]:
        """Returns the size of the effective weights"""
        return (self.num_rows, self.num_cols)

    def get_eff_block(self, row: int, col: int) -> torch.Tensor:
        """Returns a specific block detached from compute graph"""
        if len(self._weights[row][col]) == 1:
            return self._weights[row][col][0].detach()
        else:
            [u, v] = self._weights[row][col]
            block = u @ torch.diag(self._masks[row][col]) @ v
            return block.detach()

    def eff_numel(self) -> int:
        """Returns the number of parameters (accounting masking)"""
        res = 0
        for row_ind, mask_row in enumerate(self._masks):
            for col_ind, mask in enumerate(mask_row):
                if mask is None:
                    res += self._weights[row_ind][col_ind][0].numel()
                else:
                    [u, v] = self._weights[row_ind][col_ind]
                    res += (u.numel() + v.numel()) * int(mask.sum()) // mask.numel()
        return res
