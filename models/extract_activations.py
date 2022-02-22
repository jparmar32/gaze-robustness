import numpy as np
import torch
import torch.nn as nn


def remove_model_hooks(hooks):
    for hook in hooks:
        hook.remove()

### if pass in a batch, activation will be batch outputs
def get_model_activations(img,model):
    activations = []

    def hook_feature(module, input, output):
        activations.append(output)

    ## essentially want all the batch norm ones, make sure activations is of len 49 when first run
    hooks = []
    for name, layer in model.named_modules():
        if "bn" in name or "downsample.1" in name:
            if "downsample.1" in name:
                hooks[-1].remove()
                hooks.pop()

            hooks.append(layer.register_forward_hook(hook_feature))

    pred = model(img)
    remove_model_hooks(hooks)

    return activations 