import torch

def feature_squeezing(images, bits=5):
    """
    Stronger bit-depth reduction.
    Useful against bit-depth attacks.
    """

    images = torch.clamp(images, 0.0, 1.0)

    levels = 2 ** bits
    squeezed = torch.round(images * (levels - 1)) / (levels - 1)

    return squeezed