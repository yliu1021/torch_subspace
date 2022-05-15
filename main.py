import torch
from torch import nn, optim

from torch_subspace import vgg

from experiments.data import get_data
from experiments.train import train

device = torch.device("cpu")

train_data, test_data, num_classes = data = get_data("cifar10", batch_size=128)

model = vgg.vgg11(batch_norm=True, num_classes=num_classes)

loss = nn.CrossEntropyLoss()
opt = optim.SGD(
    model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4
)
train(model, train_data, loss_fn=loss, optimizer=opt, device=device)
