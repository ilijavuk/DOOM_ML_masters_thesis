from torch import nn

def initialise_network_weights(module: nn.Module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(
            module.weight,
            a=0.1,
            mode='fan_in',
            nonlinearity='leaky_relu'
        )

    for submodule in module.children():
        initialise_network_weights(submodule)
