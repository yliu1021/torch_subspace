from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# mean and std for CIFAR10 and CIFAR100 acquired from: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
loaders = {
    "cifar10": (
        datasets.CIFAR10,
        10,  # num_classes
        [0.4914, 0.4822, 0.4465],  # mean
        [0.2470, 0.2435, 0.2616],  # std
    ),
    "cifar100": (
        datasets.CIFAR100,
        100,
        [0.5071, 0.4867, 0.4408],
        [0.2675, 0.2565, 0.2761],
    ),
}


def get_data(name: str, batch_size: int = 128, data_path: str = "data"):
    if name not in loaders:
        raise ValueError(f"Invalid dataset {name}")
    data, num_classes, mean, std = loaders[name]
    normalize = transforms.Normalize(mean=mean, std=std)
    train = data(
        root=data_path,
        train=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=False,
    )
    test = data(
        root=data_path,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=False,
    )
    train = DataLoader(train, batch_size=batch_size, shuffle=True)
    test = DataLoader(test, batch_size=batch_size)
    return train, test, num_classes


if __name__ == "__main__":
    get_data("cifar100")
    get_data("cifar10")
