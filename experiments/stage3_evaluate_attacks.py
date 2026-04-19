import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from models.simple_cnn import SimpleCNN
from data.data_loader import get_dataloaders
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.deepfool import deepfool_attack
from attacks.cw import cw_attack


device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model, loader):
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total

def evaluate_attack(model, loader, attack_fn, name):
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack_fn(model, images, labels)
        with torch.no_grad():
            outputs = model(adv_images)
            preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = correct / total
    print(f"{name} Accuracy: {acc:.4f}")
    return acc

def run():
    _, test_loader = get_dataloaders("cifar10", batch_size=64)

    model = SimpleCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(
        os.path.join(PROJECT_ROOT, "saved_models", "cifar10_model.pth"),
        map_location=device))
    model.eval()

    clean_acc = evaluate(model, test_loader)
    print(f"\nClean Accuracy: {clean_acc:.4f}")

    fgsm_acc = evaluate_attack(model, test_loader, fgsm_attack, "FGSM")
    pgd_acc = evaluate_attack(model, test_loader, pgd_attack, "PGD")
    deepfool_acc = evaluate_attack(model, test_loader, deepfool_attack, "DeepFool")
    cw_acc = evaluate_attack(model, test_loader, cw_attack, "C&W")

    print("\nSummary:")
    print(f"Clean: {clean_acc:.4f}")
    print(f"FGSM: {fgsm_acc:.4f}")
    print(f"PGD: {pgd_acc:.4f}")
    print(f"DeepFool: {deepfool_acc:.4f}")
    print(f"C&W: {cw_acc:.4f}")

if __name__ == "__main__":
    run()