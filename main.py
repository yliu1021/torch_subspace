import torch
from torch import nn, optim

from torch_subspace import vgg

import pruners
from pruners import alignment

from experiments.data import get_data
from experiments.train import train, test
from torch_subspace.lr import SubspaceLR


def calc_size(model: nn.Module) -> int:
    size = 0
    for module in model.modules():
        if not isinstance(module, SubspaceLR):
            continue
        size += module.eff_numel()
    return size


def main(
    device: str,
    dataset: str,
    batch_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
):
    device = torch.device(device)
    train_data, test_data, num_classes = get_data(dataset, batch_size=batch_size)
    model = vgg.vgg11(batch_norm=True, num_classes=num_classes, device=device)

    def fit(
        epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
    ):
        loss_fn = nn.CrossEntropyLoss()
        opt = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
        train_metrics, test_metrics = [], []
        for i in range(epochs):
            print(f"Epoch: {i+1}/{epochs}")
            train_res = train(
                model, train_data, loss_fn=loss_fn, optimizer=opt, device=device
            )
            test_res = test(model, test_data, loss_fn=loss_fn, device=device)
            scheduler.step()
            train_metrics.append(train_res)
            test_metrics.append(test_res)
        return train_metrics, test_metrics

    fit(
        epochs=120,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    print(f"Preprune mem allocated: {torch.cuda.memory_allocated()}")
    preprune_size = calc_size(model)
    alignment.make_blocks(model)
    scores = alignment.prune(model, train_data=train_data, device=device)
    print(f"Post scoring mem allocated: {torch.cuda.memory_allocated()}")
    pruners.prune_scores(model, scores, sparsity=0.93, device=device)
    print(f"Postprune mem allocated: {torch.cuda.memory_allocated()}")
    postprune_size = calc_size(model)

    print(f"Preprune size: {preprune_size}")
    print(f"Postprune size: {postprune_size}")

    fit(
        epochs=60,
        lr=lr / 8,
        momentum=momentum,
        weight_decay=weight_decay
    )


if __name__ == "__main__":
    main(
        device="cuda:0",
        dataset="cifar10",
        batch_size=256,
        lr=0.05,
        momentum=0.9,
        weight_decay=5e-4,
    )
