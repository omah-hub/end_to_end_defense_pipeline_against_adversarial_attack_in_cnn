import torch.nn.functional as F
import random
import torch
def resize_smoothing(images, min_scale=0.9, max_scale=1.1):
    """
    Randomized resizing defense.
    Stronger than fixed interpolation.
    """

    scale = random.uniform(min_scale, max_scale)

    _, _, h, w = images.shape

    resized = F.interpolate(
        images,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False
    )

    restored = F.interpolate(
        resized,
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )

    return torch.clamp(restored, 0.0, 1.0)