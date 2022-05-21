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
    # Training parameters
    model_name: str,
    dataset_name: str,
    batch_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    # Pruning parameters
    target_sparsity: float,
    lr_downsize: float,
    preprune_epochs: int,
    postprune_epochs: int,
):
    device = torch.device(device)
    train_data, test_data, num_classes = get_data(dataset_name, batch_size=batch_size)
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

    best_pre_loss, best_pre_acc = fit(
        epochs=preprune_epochs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        train_type="preprune",
    )
    # TODO: save model

    print("Converting to LR model")
    torch_subspace.convert_model_to_lr(model)
    preprune_size = calc_size(model)
    print("Blocking")
    blockers.square.make_blocks(model)
    # blockers.alds.make_blocks(model, k=4)
    print("Pruning")
    # pruners.alignment_output.prune(
    #     model, train_data=train_data, sparsity=target_sparsity, device=device
    # )
    pruners.rel_error.prune(model, sparsity=target_sparsity, device=device)
    postprune_size = calc_size(model)
    print(f"Preprune size: {preprune_size}")
    print(f"Postprune size: {postprune_size}")
    eff_sparsity = 1 - postprune_size / preprune_size
    print(f"Sparsity: {eff_sparsity:.4f}")

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
        },
        metric_dict={
            "best_pre_loss": best_pre_loss,
            "best_pre_acc": best_pre_acc,
            "best_post_loss": best_post_loss,
            "best_post_acc": best_post_acc,
            "immediate_post_loss": immediate_post_loss,
            "immediate_post_acc": immediate_post_acc,
        },
    )
    writer.flush()


if __name__ == "__main__":
    main(
        device="cuda:4",
        model_name="vgg16",
        dataset_name="cifar10",
        batch_size=256,
        lr=0.05,
        momentum=0.9,
        weight_decay=5e-4,
        target_sparsity=0.95,
        lr_downsize=5,
        preprune_epochs=160,
        postprune_epochs=160,
    )
