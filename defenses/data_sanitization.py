import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def filter_high_loss_samples(model, dataloader, device, threshold_percent=0.2, batch_size=64):
    """
    Removes top X% high-loss samples from dataset and returns a DataLoader.
    
    Args:
        model: PyTorch model to compute per-sample loss.
        dataloader: DataLoader with dataset to filter.
        device: 'cuda' or 'cpu'.
        threshold_percent: Fraction of high-loss samples to remove (0.2 = remove top 20%).
        batch_size: Batch size for returned DataLoader.

    Returns:
        filtered_loader: DataLoader with high-loss samples removed.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    all_losses = []
    all_images = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_losses = criterion(outputs, labels)

            all_losses.append(batch_losses.cpu())
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())

    all_losses = torch.cat(all_losses)
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    cutoff = torch.quantile(all_losses, 1 - threshold_percent)

    mask = all_losses < cutoff
    filtered_images = all_images[mask]
    filtered_labels = all_labels[mask]
    print(f"Removed {(~mask).sum().item()} suspicious samples")

    # Wrap in DataLoader
    filtered_dataset = TensorDataset(filtered_images, filtered_labels)
    filtered_loader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)

    return filtered_loader