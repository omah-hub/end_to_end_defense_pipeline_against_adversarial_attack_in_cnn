import torch

def color_channel_attack(images, strength=0.1):
    noise = torch.randn_like(images) * strength

    attacked = images + noise
    attacked = torch.clamp(attacked, 0.0, 1.0)

    return attacked