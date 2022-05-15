import unittest

import torch

from torch_subspace import vgg


class TestVGG(unittest.TestCase):
    def test_vgg11_forward(self):
        module = vgg.vgg11(batch_norm=True, num_classes=10)
        input = torch.rand(1, 3, 64, 64, dtype=torch.float)
        output = module(input)
        self.assertEqual(output.shape, (1, 10))

    def test_vgg13_forward(self):
        module = vgg.vgg13(batch_norm=True, num_classes=10)
        input = torch.rand(1, 3, 64, 64, dtype=torch.float)
        output = module(input)
        self.assertEqual(output.shape, (1, 10))

    def test_vgg16_forward(self):
        module = vgg.vgg16(batch_norm=True, num_classes=10)
        input = torch.rand(1, 3, 64, 64, dtype=torch.float)
        output = module(input)
        self.assertEqual(output.shape, (1, 10))

    def test_vgg19_forward(self):
        module = vgg.vgg19(batch_norm=True, num_classes=10)
        input = torch.rand(1, 3, 64, 64, dtype=torch.float)
        output = module(input)
        self.assertEqual(output.shape, (1, 10))
