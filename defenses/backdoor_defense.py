import torch
import torch.nn as nn


def fine_pruning(model, dataloader, device, prune_percent=0.5):
    """
    Removes neurons with low average activation.
    """

    model.eval()

    activations = []

    def hook(module, input, output):
        activations.append(output.detach())

    # Attach hook to last layer
    handle = model.fc1.register_forward_hook(hook)

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _ = model(images)

    handle.remove()

    # Compute average activation
    avg_activation = torch.mean(torch.cat(activations), dim=0)

    # Prune lowest activations (simple version)
    threshold = torch.quantile(avg_activation, prune_percent)

    # You can extend this to zero-out weights if needed

    return model