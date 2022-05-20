import unittest

from torch import nn
from torchvision import models

from torch_subspace.convert import convert_model_to_lr


class TestConvert(unittest.TestCase):
    def test_resnet(self):
        model = models.resnet18()
        convert_model_to_lr(model)
        self.assertTrue(
            all(
                not isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)
                for layer in model.modules()
            )
        )
