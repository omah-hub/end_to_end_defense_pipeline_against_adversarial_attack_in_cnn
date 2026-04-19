import torch
import torch.nn.functional as F


def compute_entropy(predictions):

    probs = F.softmax(predictions, dim=1)
    log_probs = torch.log(probs + 1e-8)

    entropy = -torch.sum(probs * log_probs, dim=1)

    return entropy.mean()


def strip_detect(model, image, clean_images, device, num_samples=10):

    model.eval()

    entropies = []

    for i in range(num_samples):

        random_img = clean_images[i].to(device)

        # Mix images
        alpha = torch.rand(1).item()
        mixed = alpha * image + (1 - alpha) * random_img
        mixed = mixed.unsqueeze(0)

        with torch.no_grad():
            output = model(mixed)

        entropy = compute_entropy(output)

        entropies.append(entropy.item())

    return sum(entropies) / len(entropies)


def strip_defense(model, data_loader, clean_loader, device, threshold=2.0, max_check=2000):

    clean_images = []

    for images, _ in clean_loader:
        clean_images.extend(images)
        if len(clean_images) > 100:
            break

    suspicious_indices = []
    index = 0  # ✅ initialize here

    for images, _ in data_loader:

        images = images.to(device)

        for img in images:

            # Stop early if max reached
            if index >= max_check:
                detection_rate = len(suspicious_indices) / max_check
                return suspicious_indices, detection_rate

            entropy = strip_detect(model, img, clean_images, device)

            if entropy < threshold:
                suspicious_indices.append(index)

            index += 1

    # If loop finishes normally
    detection_rate = len(suspicious_indices) / index

    return suspicious_indices, detection_rate