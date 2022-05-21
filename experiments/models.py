from torchvision import models

all_models = {
    "vgg11": models.vgg11_bn,
    "vgg13": models.vgg13_bn,
    "vgg16": models.vgg16_bn,
    "vgg19": models.vgg19_bn,
}


def get_model(model_name: str, num_classes: int, device=None):
    if model_name in all_models:
        return all_models[model_name](num_classes=num_classes).to(device=device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
