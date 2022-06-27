from typing import Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def calc_num_correct(y_pred, y_true):
    return (y_pred.argmax(1) == y_true).type(torch.float).sum().item()


def train(
    model: nn.Module,
    data: DataLoader,
    loss_fn,
    optimizer: optim.Optimizer,
    device,
):
    size = len(data.dataset)
    model.train()
    step = 0
    losses = []
    accuracies = []
    for X, y in data:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += len(X)
        losses.append(loss.item())
        accuracies.append(calc_num_correct(pred, y) / len(y))
        print(
            f"\r[Train] Step: {step:>05} / {size} | Loss: {np.mean(losses):.5f} | Accuracy: {np.mean(accuracies):.4f}",
            end="",
        )
    print()
    return np.mean(losses), np.mean(accuracies)


def test(model: nn.Module, data: DataLoader, loss_fn, device, verbose=True):
    size = len(data.dataset)
    num_batches = len(data)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if verbose:
        print(f"[Test] Loss: {test_loss:.5f} | Accuracy: {correct:.4f}")
    return test_loss, correct
