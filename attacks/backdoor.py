import torch
import random


def add_trigger(images, trigger_size=4, position="bottom_right", trigger_value=1.0):
    """
    Adds a simple square trigger to a batch of images.

    Args:
        images: Tensor (B, C, H, W)
        trigger_size: size of trigger square
        position: where to place trigger
        trigger_value: pixel value of trigger (1.0 = white)
    """

    images = images.clone()

    B, C, H, W = images.shape

    for i in range(B):

        if position == "bottom_right":
            images[i, :, H-trigger_size:H, W-trigger_size:W] = trigger_value

        elif position == "top_left":
            images[i, :, 0:trigger_size, 0:trigger_size] = trigger_value

    return images


def backdoor_poison(
    images,
    labels,
    target_class=0,
    poison_rate=0.1,
    trigger_size=4,
    position="bottom_right"
):
    """
    Backdoor attack:
    - Randomly poisons a percentage of samples
    - Adds trigger
    - Changes label to target_class
    """

    images = images.clone()
    labels = labels.clone()

    for i in range(len(images)):

        if random.random() < poison_rate:
            # Add trigger
            images[i:i+1] = add_trigger(
                images[i:i+1],
                trigger_size=trigger_size,
                position=position
            )

            # Change label to target class
            labels[i] = target_class

    return images, labels