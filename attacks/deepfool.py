import torch
import torch.nn.functional as F

def deepfool_attack(model, images, labels, epsilon=0.02):
    images = images.clone().detach().requires_grad_(True)

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()

    grad = images.grad
    adv_images = images + epsilon * grad
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()