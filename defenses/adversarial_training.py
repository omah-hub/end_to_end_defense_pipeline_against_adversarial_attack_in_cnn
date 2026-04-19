import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from models.simple_cnn import SimpleCNN
from data.data_loader import get_dataloaders
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_adversarial_model(epochs=5, epsilon=0.03, alpha=0.007, attacks=["fgsm", "pgd"]):

    train_loader, _ = get_dataloaders("cifar10", batch_size=64)

    model = SimpleCNN(num_classes=10).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"Training robust model (clean + multi-attack: {attacks})...")

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Randomly pick an attack per batch
            attack_type = random.choice(attacks)

            if attack_type == "fgsm":
                adv_images = fgsm_attack(model, images, labels, epsilon=epsilon)
            elif attack_type == "pgd":
                adv_images = pgd_attack(model, images, labels, epsilon=epsilon, alpha=alpha, iters=5)  # reduce PGD iters for speed
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")

            # Combine clean + adversarial images
            combined_images = torch.cat([images, adv_images], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)

            outputs = model(combined_images)
            loss = criterion(outputs, combined_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # Save robust model
    save_path = os.path.join(PROJECT_ROOT, "saved_models", "simple_cnn_robust_multi_attack.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Robust model saved at {save_path}")

if __name__ == "__main__":
    train_adversarial_model(epochs=30, attacks=["fgsm", "pgd"])