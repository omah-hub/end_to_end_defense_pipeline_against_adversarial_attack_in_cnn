import torch
from torch.utils.data import TensorDataset, DataLoader


def spectral_signature_filter(model, dataloader, device, remove_ratio=0.1):

    model.eval()

    all_features = []
    all_images = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            _, features = model(images, return_features=True)

            all_features.append(features.cpu())
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())

    all_features = torch.cat(all_features)
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    # Center
    mean = all_features.mean(dim=0, keepdim=True)
    centered = all_features - mean

    # SVD
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    top_vector = Vh[0]

    # Scores
    scores = torch.matmul(centered, top_vector)

    # Remove top outliers
    num_remove = int(remove_ratio * len(scores))

    _, indices = torch.topk(scores.abs(), num_remove)

    mask = torch.ones(len(scores), dtype=torch.bool)
    mask[indices] = False

    clean_images = all_images[mask]
    clean_labels = all_labels[mask]

    clean_dataset = TensorDataset(clean_images, clean_labels)

    return DataLoader(clean_dataset, batch_size=dataloader.batch_size, shuffle=True)