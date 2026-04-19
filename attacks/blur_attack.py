import torch
import torchvision.transforms.functional as TF

def gaussian_blur_attack(images, kernel_size=5):
    blurred_images = []
    
    for img in images:
        blurred = TF.gaussian_blur(img, kernel_size)
        blurred_images.append(blurred)

    return torch.stack(blurred_images).to(images.device)