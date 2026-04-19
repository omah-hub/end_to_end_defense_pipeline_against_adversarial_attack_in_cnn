import torch

def bit_depth_reduction_attack(images, bits=4):
    levels = 2 ** bits
    images = torch.floor(images * levels) / levels
    images = torch.clamp(images, 0.0, 1.0)
    return images