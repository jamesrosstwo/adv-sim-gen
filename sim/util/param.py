from torch import nn


def randomize_weights(module):
    """
    Apply random weight initialization to specific layer types.

    - ConvTranspose2d: Weights initialized with normal distribution (mean=0.0, std=0.02).
    - Conv2d: Weights initialized with normal distribution (mean=0.0, std=0.02).
    - Linear: Weights initialized with normal distribution (mean=0.0, std=0.02).
    Biases are initialized to 0.0 if they exist for these layers.
    """
    if isinstance(module, nn.ConvTranspose2d):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
