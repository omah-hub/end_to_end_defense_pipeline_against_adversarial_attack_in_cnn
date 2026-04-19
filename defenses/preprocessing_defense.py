import torch
import torchvision.transforms.functional as TF

def denoise_images(images, method="clip", clip_min=0.0, clip_max=1.0):

    if method == "clip":
        return torch.clamp(images, clip_min, clip_max)

    elif method == "smoothing":
        smoothed = []
        for img in images:
            smoothed.append(TF.gaussian_blur(img, kernel_size=3))
        return torch.stack(smoothed)

    elif method == "sharpen":
        sharpened = []
        for img in images:
            blurred = TF.gaussian_blur(img, kernel_size=3)
            sharp = img + (img - blurred)  # unsharp masking
            sharpened.append(torch.clamp(sharp, 0, 1))
        return torch.stack(sharpened)

    return images