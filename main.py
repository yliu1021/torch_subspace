import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

import blockers
import pruners
import torch_subspace
from experiments.data import get_data
from experiments.models import get_model
from experiments.train import test, train
from torch_subspace.lr import SubspaceLR


def calc_size(model: nn.Module) -> int:
    size = 0
    for module in model.modules():
        if not isinstance(module, SubspaceLR):
            continue
        if not module.is_leaf:
            continue
        size += module.numels()
    return size


def main(
    device: str,
    data_location: str,
    save_path: Optional[str],
    # Training parameters
    model_name: str,
    dataset_name: str,
    batch_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    # Pruning parameters
    blocker_name: str,
    pruner_name: str,
    target_sparsity: float,
    lr_downsize: float,
    preprune_epochs: int,
    postprune_epochs: int,
):
    if save_path is not None:
        save_path = Path(save_path)
    device = torch.device(device)
    train_data, test_data, num_classes = get_data(
        dataset_name, batch_size=batch_size, data_path=data_location
    )
    model = get_model(model_name, num_classes=num_classes, device=device)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    def fit(
        epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        train_type: str,  # preprune or post prune
    ):
        opt = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
        best_train_loss, best_test_loss = float("inf"), float("inf")
        best_train_acc, best_test_acc = float("-inf"), float("-inf")
        for i in range(1, epochs + 1):
            print(f"Epoch: {i}/{epochs}")
            train_res = train(
                model, train_data, loss_fn=loss_fn, optimizer=opt, device=device
            )
            test_res = test(model, test_data, loss_fn=loss_fn, device=device)
            scheduler.step()
            writer.add_scalars(
                f"Loss/{train_type}",
                tag_scalar_dict={"train": train_res[0], "test": test_res[0]},
                global_step=i,
            )
            writer.add_scalars(
                f"Accuracy/{train_type}",
                tag_scalar_dict={"train": train_res[1], "test": test_res[1]},
                global_step=i,
            )
            best_train_loss = min(best_train_loss, train_res[0])
            best_test_loss = min(best_test_loss, test_res[0])
            best_train_acc = max(best_train_acc, train_res[1])
            best_test_acc = max(best_test_acc, test_res[1])
        writer.flush()
        return best_test_loss, best_test_acc

    if save_path is not None and save_path.exists():
        model.load_state_dict(torch.load(save_path))
        print("Loaded saved model")
        best_pre_loss, best_pre_acc = test(model, test_data, loss_fn=loss_fn, device=device)
    else:
        # Warmup
        opt = optim.SGD(
            model.parameters(), lr=0, momentum=momentum, weight_decay=weight_decay
        )
        for lr in np.linspace(0, lr, num=11, endpoint=True)[1:]:
            for g in opt.param_groups:
                g["lr"] = lr
            print(f"Warmup lr: {lr}")
            train(model, train_data, loss_fn=loss_fn, optimizer=opt, device=device)
        # Fit for `preprune_epochs` epochs
        best_pre_loss, best_pre_acc = fit(
            epochs=preprune_epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            train_type="preprune",
        )
        if save_path is not None:
            torch.save(model.state_dict(), save_path)

    # Capture a baseline output for comparison post pruning
    sample_in, _ = next(iter(train_data))
    sample_in = sample_in.to(device)
    with torch.no_grad():
        baseline_output = model(sample_in)

    print("Converting to LR model")
    torch_subspace.convert_model_to_lr(model)
    test(model, test_data, loss_fn=loss_fn, device=device)
    preprune_size = calc_size(model)
    print("Blocking")
    if blocker_name == "square":
        blockers.square.make_blocks(model)
    elif blocker_name == "alds":
        blockers.alds.make_blocks(model, k=4)  # TODO: tune `k` value
    elif blocker_name == "none":
        pass
    else:
        raise ValueError(f"Invalid blocker: {blocker_name}")
    print("Pruning")
    if pruner_name == "alignment_output":
        pruners.alignment_output.prune(
            model, train_data=train_data, sparsity=target_sparsity, device=device
        )
    elif pruner_name == "alignment_output_sampling":
        pruners.alignment_output_sampling.prune(
            model,
            train_data=train_data,
            sparsity=target_sparsity,
            proportional_sampling=False,
            device=device,
        )
    elif pruner_name == "alignment_output_sampling_proportional":
        pruners.alignment_output_sampling.prune(
            model,
            train_data=train_data,
            sparsity=target_sparsity,
            proportional_sampling=True,
            device=device,
        )
    elif pruner_name == "alignment_variance":
        pruners.alignment_variance.prune(
            model,
            train_data=train_data,
            sparsity=target_sparsity,
            device=device,
        )
    elif pruner_name == "relative_error":
        pruners.rel_error.prune(model, sparsity=target_sparsity, device=device)
    elif pruner_name == "magnitude":
        pruners.magnitude.prune(model, sparsity=target_sparsity, device=device)
    else:
        raise ValueError(f"Invalid pruner: {pruner_name}")
    postprune_size = calc_size(model)
    print(f"Preprune size: {preprune_size}")
    print(f"Postprune size: {postprune_size}")
    eff_sparsity = 1 - postprune_size / preprune_size
    print(f"Sparsity: {eff_sparsity:.4f}")

    # For comparing against the baseline output
    with torch.no_grad():
        pruned_output = model(sample_in)
        alignment_score = torch.mean((baseline_output - pruned_output) ** 2).cpu()

    print("Post prune evaluation...")
    immediate_post_loss, immediate_post_acc = test(
        model, test_data, loss_fn=loss_fn, device=device
    )

    print("Finetuning...")
    best_post_loss, best_post_acc = fit(
        epochs=postprune_epochs,
        lr=lr / lr_downsize,
        momentum=momentum,
        weight_decay=weight_decay,
        train_type="postprune",
    )
    postprune_size_check = calc_size(model)
    assert postprune_size == postprune_size_check

    writer.add_hparams(
        hparam_dict={
            "dataset": dataset_name,
            "model": model_name,
            "batch_size": batch_size,
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "sparsity": eff_sparsity,
            "lr_downsize": lr_downsize,
            "preprune_epochs": preprune_epochs,
            "postprune_epochs": postprune_epochs,
            "preprune_size": preprune_size,
            "postprune_size": postprune_size,
            "blocker_name": blocker_name,
            "pruner_name": pruner_name,
        },
        metric_dict={
            "best_pre_loss": best_pre_loss,
            "best_pre_acc": best_pre_acc,
            "best_post_loss": best_post_loss,
            "best_post_acc": best_post_acc,
            "immediate_post_loss": immediate_post_loss,
            "immediate_post_acc": immediate_post_acc,
            "alignment_score": alignment_score,
        },
    )
    writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=str,
        choices=["cpu"] + ["cuda"] + [f"cuda:{x}" for x in range(8)],
        default="cuda",
    )
    parser.add_argument("--data_location", type=str, default="data")
    parser.add_argument(
        "--model", type=str, choices=["vgg11", "vgg16", "vgg19"], required=True
    )
    parser.add_argument("--save_path", type=str)
    parser.add_argument(
        "--dataset", type=str, choices=["cifar10", "cifar100"], required=True
    )
    parser.add_argument(
        "--blocker", type=str, choices=["square", "alds", "none"], required=True
    )
    parser.add_argument(
        "--pruner",
        type=str,
        choices=[
            "alignment_output",
            "alignment_output_sampling",
            "alignment_output_sampling_proportional",
            "alignment_variance",
            "relative_error",
            "magnitude",
        ],
        required=True,
    )
    parser.add_argument("--sparsity", type=float, required=True)
    parser.add_argument("--preprune_epochs", type=int, default=160)
    parser.add_argument("--postprune_epochs", type=int, default=160)
    args = parser.parse_args()

    main(
        device=args.gpu,
        data_location=args.data_location,
        save_path=args.save_path,
        model_name=args.model,
        dataset_name=args.dataset,
        batch_size=256,
        lr=0.05,
        momentum=0.9,
        weight_decay=5e-4,
        blocker_name=args.blocker,
        pruner_name=args.pruner,
        target_sparsity=args.sparsity,
        lr_downsize=4,
        preprune_epochs=args.preprune_epochs,
        postprune_epochs=args.postprune_epochs,
    )
