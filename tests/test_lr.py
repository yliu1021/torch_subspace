import unittest

import torch
from torch import nn

from torch_subspace import SubspaceLR


class TestSubspaceLR(unittest.TestCase):
    def setUp(self) -> None:
        weights = torch.randn(10, 7)
        self.module = SubspaceLR(num_rows=10, num_cols=7)
        self.module.set_eff_weights(weights)
        self.initial_eff_weights = self.module.eff_weights().detach()

    def assertSameAsOriginal(self):
        self.assertAlmostEqual(
            torch.norm(self.initial_eff_weights - self.module.eff_weights()).item(),
            0,
            places=4,
        )

    def test_is_leaf_initially(self):
        self.assertTrue(self.module.is_leaf)

    def test_is_leaf_after_split(self):
        self.module.split([3, 4], direction="horizontal")
        self.assertEqual(len(self.module.weights), 2)
        self.assertIsInstance(self.module.weights, nn.ModuleList)
        self.assertFalse(self.module.is_leaf)

    def test_set_mask(self):
        self.module.set_mask(torch.ones(self.module.max_rank()))
        self.assertEqual(len(self.module.weights), 2)
        self.assertIsInstance(self.module.weights, nn.ParameterList)
        self.assertSameAsOriginal()

    def test_clear_mask(self):
        self.module.set_mask(torch.ones(self.module.max_rank()))
        self.module.clear_mask()
        self.assertEqual(len(self.module.weights), 1)
        self.assertIsInstance(self.module.weights, nn.ParameterList)
        self.assertSameAsOriginal()

    def test_set_eff_weights(self):
        new_weights = torch.randn(10, 7)
        self.module.set_eff_weights(new_weights)
        self.assertAlmostEqual(
            torch.norm(new_weights - self.module.eff_weights()).item(),
            0,
            places=4,
        )
        self.module.set_mask(torch.ones(7))
        new_weights = torch.randn(10, 7)
        self.module.set_eff_weights(new_weights)
        self.assertAlmostEqual(
            torch.norm(new_weights - self.module.eff_weights()).item(),
            0,
            places=4,
        )
        self.module.split([2, 5], direction="horizontal")
        self.module.set_eff_weights(self.initial_eff_weights)
        self.assertSameAsOriginal()

    def test_max_rank(self):
        self.assertEqual(self.module.max_rank(), 7)

    def test_split(self):
        self.module.split([3, 7], direction="vertical")
        self.assertEqual(len(self.module.weights), 2)
        self.assertIsInstance(self.module.weights, nn.ModuleList)
        self.assertEqual(self.module.weights[0].shape, (3, 7))
        self.assertEqual(self.module.weights[1].shape, (7, 7))
        self.assertEqual(self.module.direction, "vertical")
        self.assertSameAsOriginal()

    def test_collect(self):
        self.module.split([3, 7], direction="vertical")
        self.module.collect()
        self.assertEqual(len(self.module.weights), 1)
        self.assertIsInstance(self.module.weights, nn.ParameterList)
        self.assertSameAsOriginal()

    def test_numels(self):
        self.assertEqual(self.module.numels(), 70)
        self.module.split([2, 8], direction="vertical")
        self.assertEqual(self.module.numels(), 70)
        self.module.weights[0].set_mask(torch.tensor([1.0, 0.0]))
        self.assertEqual(self.module.numels(), 65)

    def test_shape(self):
        self.assertEqual(self.module.shape, (10, 7))
