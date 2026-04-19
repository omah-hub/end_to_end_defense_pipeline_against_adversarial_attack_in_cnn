import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from data.data_loader import get_dataloaders
from models.simple_cnn import SimpleCNN
from attacks.poison import label_flip  # Example poisoning attack
from evaluation.metrics import evaluate_model, accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_poisoned_model(poison_rate=0.1, epochs=10, batch_size=64, lr=1e-3):
    # -----------------------
    # Load data
    # -----------------------
    train_loader, test_loader = get_dataloaders("cifar10", batch_size=batch_size)

    # -----------------------
    # Initialize model
    # -----------------------
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # -----------------------
    # Apply label flipping for poisoning
    # -----------------------
    print(f"Applying label flip with rate {poison_rate}")
   # -----------------------
# Apply label flipping properly (ONCE)
# -----------------------
    print(f"Applying label flip with rate {poison_rate}")

# Convert targets to tensor
    targets = torch.tensor(train_loader.dataset.targets)

# Apply poisoning
    poisoned_targets = label_flip(targets, poison_rate=poison_rate)

# Convert back to list (VERY IMPORTANT)
    train_loader.dataset.targets = poisoned_targets.tolist()

    # -----------------------
    # Training
    # -----------------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # -----------------------
        # Evaluate on clean test set
        # -----------------------
        clean_acc = evaluate_model(model, test_loader, device)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} | Clean Accuracy: {clean_acc:.4f}")

    # -----------------------
    # Save poisoned-trained model
    # -----------------------
    save_path = os.path.join(PROJECT_ROOT, "saved_models", "cifar10_poisoned.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Poisoned model saved at: {save_path}")

    # -----------------------
    # Final evaluation
    # -----------------------
    print("Evaluating poisoned model performance:")
    final_acc = evaluate_model(model, test_loader, device)
    print(f"Final Clean Accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    train_poisoned_model()