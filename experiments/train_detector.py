import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from models.simple_cnn import SimpleCNN
from data.data_loader import get_dataloaders
from attacks.pgd import pgd_attack
from defenses.detector import Detector

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_detector():
    # -----------------------------
    # Load robust classifier (PGD)
    # -----------------------------
    classifier = SimpleCNN(num_classes=10).to(device)
    classifier_path = os.path.join(PROJECT_ROOT, "saved_models", "simple_cnn_robust_multi_attack.pth")
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()

    # Freeze classifier
    for p in classifier.parameters():
        p.requires_grad = False

    # -----------------------------
    # Create detector
    # -----------------------------
    detector = Detector(input_dim=10).to(device)
    optimizer = Adam(detector.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    train_loader, _ = get_dataloaders("cifar10", batch_size=64)
    epochs = 30

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Clean logits
            with torch.no_grad():
                clean_logits = classifier(images)
            clean_targets = torch.zeros(images.size(0), device=device)

            # Adversarial logits (PGD attack)
            adv_images = pgd_attack(classifier, images, labels, epsilon=0.03, alpha=0.007, iters=10)
            with torch.no_grad():
                adv_logits = classifier(adv_images)
            adv_targets = torch.ones(images.size(0), device=device)

            # Combine
            logits = torch.cat([clean_logits, adv_logits], dim=0)
            targets = torch.cat([clean_targets, adv_targets], dim=0).unsqueeze(1)  # shape (batch_size*2, 1)

            # Train detector
            preds = detector(logits)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # Save detector inside detector folder
    save_path = os.path.join(PROJECT_ROOT, "saved_models", "detector_cifar10.pth")
    torch.save(detector.state_dict(), save_path)
    print("Detector saved at:", save_path)

if __name__ == "__main__":
    train_detector()