import torch
import torch.nn.functional as F

def resize_attack(images, scale=0.8):

    b, c, h, w = images.shape
    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = F.interpolate(images, size=(new_h, new_w), mode="bilinear")
    restored = F.interpolate(resized, size=(h, w), mode="bilinear")

    return torch.clamp(restored, 0.0, 1.0)