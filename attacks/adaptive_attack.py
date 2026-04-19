# attacks/adaptive_attack.py

import torch
import random


def adaptive_label_flip(dataset, poison_rate=0.2, num_classes=10):
    """
    Adaptive Label Flipping Attack.
    Strategy:
    - Flip labels carefully
    - Avoid extremely random patterns
    - Make poisoning less detectable
    """

    images, labels = dataset.tensors

    num_poison = int(len(labels) * poison_rate)

    # Select random indices
    indices = random.sample(range(len(labels)), num_poison)

    for i in indices:
        # Instead of fully random label,
        # shift label by +1 (more structured poisoning)
        labels[i] = (labels[i] + 1) % num_classes

    return torch.utils.data.TensorDataset(images, labels)


def adaptive_backdoor(images, labels, target_label=0, poison_rate=0.2):
    """
    Adaptive Backdoor Attack.
    Strategy:
    - Use small invisible trigger
    - Blend trigger with image
    - Avoid large obvious patches
    """

    num_poison = int(len(labels) * poison_rate)

    for i in range(num_poison):
        # Small subtle trigger (bottom-right corner)
        images[i, :, -2:, -2:] = images[i, :, -2:, -2:] * 0.7 + 0.3

        labels[i] = target_label

    return images, labels


def adaptive_combined_attack(images, labels, poison_rate=0.2, target_label=0):

    images = images.clone()
    labels = labels.clone()

    num_poison = int(len(labels) * poison_rate)

    # ---- Backdoor ----
    for i in range(num_poison):
        images[i, :, -2:, -2:] = (
            images[i, :, -2:, -2:] * 0.7 + 0.3
        )
        labels[i] = target_label

    # ---- Structured Label Shift ----
    for i in range(num_poison):
        labels[i] = (labels[i] + 1) % 10

    return images, labels