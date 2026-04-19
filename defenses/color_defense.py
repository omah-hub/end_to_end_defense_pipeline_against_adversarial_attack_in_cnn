import torch

def color_normalization(images):

    mean = images.mean(dim=(2,3), keepdim=True)
    std = images.std(dim=(2,3), keepdim=True) + 1e-6

    normalized = (images - mean) / std

    # More stable clamping
    return torch.tanh(normalized)