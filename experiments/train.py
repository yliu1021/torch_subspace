from typing import Optional

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def calc_num_correct(y_pred, y_true):
    return (y_pred.argmax(1) == y_true).type(torch.float).sum().item()


def train(
    model: nn.Module,
    train: DataLoader,
    loss_fn,
    optimizer: optim.Optimizer,
    device,
):
    size = len(train.dataset)
    model.train()
    losses = []
    accuracies = []
    for batch, (X, y) in enumerate(train):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step = batch * len(X)
        losses.append(loss.item())
        accuracies.append(calc_num_correct(pred, y) / len(y))
        print(
            f"\rStep: {step:>05} / {size} | Loss: {np.mean(losses):.5f} | Accuracy: {np.mean(accuracies):.4f}",
            end="",
        )
    print()
    return {"loss": np.mean(losses), "accuracy": np.mean(accuracies)}


def test(model: nn.Module, test: DataLoader, loss_fn, device):
    size = len(test.dataset)
    num_batches = len(test)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return 100 * correct, test_loss
