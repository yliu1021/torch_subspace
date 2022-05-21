from torch import nn

from torch_subspace.modules import Conv2dLR, LinearLR


def convert_linear_to_lr(linear: nn.Linear) -> LinearLR:
    lr = LinearLR(linear.in_features, linear.out_features, linear.bias is not None)
    lr.set_eff_weights(linear.weight.detach())
    if linear.bias is not None:
        lr.bias = nn.Parameter(linear.bias.detach().clone())
    return lr


def convert_conv2d_to_lr(conv: nn.Conv2d) -> Conv2dLR:
    lr = Conv2dLR(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        conv.bias is not None,
        conv.padding_mode,
    )
    lr.set_eff_weights(conv.weight.detach().reshape(lr.num_rows, lr.num_cols))
    if conv.bias is not None:
        lr.bias = nn.Parameter(conv.bias.detach().clone())
    return lr


def convert_model_to_lr(model: nn.Module):
    """
    Recursively modifies a module in place to replace instances of conv2d and linear modules into
    low rank alternatives
    :param model: the module to convert
    :return:
    """
    if isinstance(model, nn.Linear):
        return convert_linear_to_lr(model)
    elif isinstance(model, nn.Conv2d):
        return convert_conv2d_to_lr(model)

    for attr_name in dir(model):
        attr = getattr(model, attr_name)
        if isinstance(attr, nn.ModuleList) or isinstance(attr, nn.Sequential):
            for i in range(len(attr)):
                attr[i] = convert_model_to_lr(attr[i])
        elif isinstance(attr, nn.ModuleDict):
            for key in attr.keys():
                attr[key] = convert_model_to_lr(attr[key])
        elif isinstance(attr, nn.Module):
            attr = convert_model_to_lr(attr)
        else:
            continue
        setattr(model, attr_name, attr)

    assert not any(
        [
            isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)
            for layer in model.modules()
        ]
    ), "Convert to lr model failed."

    return model
