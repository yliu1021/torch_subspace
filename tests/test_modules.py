import unittest

import torch

from torch_subspace import Conv2dLR, LinearLR


class TestLinearLR(unittest.TestCase):
    def setUp(self) -> None:
        self.module = LinearLR(in_features=10, out_features=8)

    def test_forward(self):
        x = torch.rand(1, 10, dtype=torch.float)
        y = self.module(x)
        self.assertTrue(y.requires_grad)
        self.assertEqual(y.shape, (1, 8))


class TestConv2dLR(unittest.TestCase):
    def setUp(self) -> None:
        self.module = Conv2dLR(
            in_channels=10, out_channels=8, kernel_size=3, padding="same"
        )

    def test_forward(self):
        x = torch.rand(1, 10, 20, 18, dtype=torch.float)
        y = self.module(x)
        self.assertTrue(y.requires_grad)
        self.assertEqual(y.shape, (1, 8, 20, 18))
