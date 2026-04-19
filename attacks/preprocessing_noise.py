import torch

def add_gaussian_noise(images, mean=0.0, std=0.05):
    noise = torch.randn_like(images) * std + mean
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0.0, 1.0)  # keep pixel values valid
    return noisy_images