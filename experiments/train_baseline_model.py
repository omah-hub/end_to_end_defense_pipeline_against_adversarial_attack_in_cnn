import sys
import os

PROJECT_ROOT = "/content/drive/MyDrive/Adversarial_Defense_Project"
sys.path.append(PROJECT_ROOT)
import torch
import torch.nn as nn
import torch.optim as optim
from data.data_loader import get_dataloaders
from models.simple_cnn import SimpleCNN
from evaluation.metrics import clean_accuracy



def train(dataset_name, num_classes, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = SimpleCNN(num_classes).to(device)

    train_loader, test_loader = get_dataloaders(dataset_name, batch_size=64)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

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

        print(f"{dataset_name} | Epoch {epoch+1} | Loss: {total_loss:.2f}")
    save_path = os.path.join(PROJECT_ROOT, "saved_models", "cifar10_model.pth")
    print(f"Saved {dataset_name} model")

    # Evaluate and print accuracy
    accuracy = clean_accuracy(model, test_loader, device)
    print(f"Clean Accuracy (from train_baseline_model): {accuracy:.4f}")

if __name__ == "__main__":
    train("cifar10", num_classes=10)
  
