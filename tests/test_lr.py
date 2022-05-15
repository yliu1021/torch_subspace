import unittest

import torch

from torch_subspace import SubspaceLR


class TestSubspaceLR(unittest.TestCase):
    def setUp(self) -> None:
        self.module = SubspaceLR(
            num_rows=10, num_cols=7, dtype=torch.float, device=torch.device("cpu")
        )
        self.initial_eff_weights = self.module.eff_weights().detach()

    def test_size(self):
        self.assertEqual(self.module.size(), (10, 7))

    def test_eff_weights_size(self):
        self.assertEqual(self.initial_eff_weights.size(), (10, 7))

    def test_set_blocks(self):
        self.module.set_blocks([2, 8], [3, 4])
        self.assertAlmostEqual(
            float(torch.sum(self.initial_eff_weights - self.module.eff_weights())), 0
        )

    def test_set_blocks_shape(self):
        self.module.set_blocks([2, 8], [3, 4])
        self.assertEqual(self.module.block_shape, (2, 2))

    def test_set_blocks_size(self):
        self.module.set_blocks([2, 8], [3, 4])
        self.assertEqual(self.module.block_size(0, 0), (2, 3))

    def test_set_mask(self):
        self.module.set_blocks([2, 8], [3, 4])
        self.module.set_mask(0, 0, torch.tensor([1, 0]))  # set a low rank mask
        self.module.set_mask(0, 0, torch.tensor([1, 1]))  # then set full rank
        # weights should be the same because the mask doesn't change the weights
        self.assertAlmostEqual(
            float(torch.sum(self.initial_eff_weights - self.module.eff_weights())),
            0,
            places=6,  # relax precision from 7 to 6 bc we're using 32 bit floats
        )

    def test_clear_mask(self):
        self.module.set_blocks([2, 8], [3, 4])
        self.module.set_mask(0, 0, torch.tensor([1, 1]))
        self.module.clear_mask(0, 0)
        self.assertAlmostEqual(
            float(torch.sum(self.initial_eff_weights - self.module.eff_weights())),
            0,
            places=6,  # relax precision from 7 to 6 bc we're using 32 bit floats
        )

    def test_block_size_lr(self):
        self.module.set_blocks([2, 8], [3, 4])
        self.module.set_mask(0, 0, torch.tensor([1, 1]))
        self.assertEqual(self.module.block_size(0, 0), (2, 3))

    def test_set_block_eff_weights(self):
        self.module.set_blocks([2, 8], [3, 4])
        self.module.set_mask(0, 0, torch.tensor([1, 1]))
        init_eff_block = self.module.get_eff_block(0, 0)
        self.module.clear_mask(0, 0)
        self.assertAlmostEqual(
            float(torch.sum(init_eff_block - self.module.get_eff_block(0, 0))),
            0,
            places=6,
        )

    def test_eff_num_el(self):
        self.assertAlmostEqual(self.module.eff_numel(), 70)
        self.module.set_blocks([2, 8], [3, 4])
        self.module.set_mask(0, 0, torch.tensor([1, 0]))
        self.assertAlmostEqual(self.module.eff_numel(), 69)
