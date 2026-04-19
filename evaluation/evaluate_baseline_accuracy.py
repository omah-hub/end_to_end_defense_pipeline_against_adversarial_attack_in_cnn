import os
import sys

PROJECT_ROOT = "/content/drive/MyDrive/Adversarial_Defense_Project"
sys.path.append(PROJECT_ROOT)
from metrics import evaluate_model
from data.data_loader import get_dataloaders
from models.simple_cnn import SimpleCNN
import torch

device = "cpu"

model = SimpleCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("saved_models/cifar10_model.pth", map_location=device))

_, cifar10_test = get_dataloaders("cifar10", batch_size=64)

clean_acc = evaluate_model(model, cifar10_test, device)
print("Clean CIFAR-10 accuracy:", clean_acc)
