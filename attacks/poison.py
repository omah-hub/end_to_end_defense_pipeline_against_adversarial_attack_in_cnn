import torch

def label_flip(labels, poison_rate=0.1, num_classes=10):
    """
    Randomly flips a percentage of labels.
    """
    device = labels.device  # Get device (cpu or cuda)

    labels = labels.clone()

    # Create mask on same device
    mask = torch.rand(labels.size(), device=device) < poison_rate

    # Create random labels on same device
    random_labels = torch.randint(
        0, num_classes, labels.size(), device=device
    )

    labels[mask] = random_labels[mask]

    return labels