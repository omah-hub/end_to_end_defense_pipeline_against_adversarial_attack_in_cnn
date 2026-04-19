import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from models.simple_cnn import SimpleCNN
from data.data_loader import get_dataloaders

# Attacks
from attacks.preprocessing_noise import add_gaussian_noise
from attacks.bit_depth_attack import bit_depth_reduction_attack
from attacks.blur_attack import gaussian_blur_attack
from attacks.resize_attack import resize_attack
from attacks.color_attack import color_channel_attack

# Defenses
from defenses.preprocessing_defense import denoise_images
from defenses.feature_squeezing import feature_squeezing
from defenses.resize_defense import resize_smoothing
from defenses.color_defense import color_normalization

# Metrics
from evaluation.metrics import (
    clean_accuracy,
    defense_gain
)
device = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# TRAIN FUNCTION
# =====================================================
def train(model, dataloader, epochs=5):
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    return model


# =====================================================
# EVALUATION FUNCTIONS
# =====================================================
def evaluate_attack(model, dataloader, attack_fn):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            attacked = attack_fn(images)
            preds = model(attacked).argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def evaluate_defense(model, dataloader, attack_fn, defense_fn):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            attacked = attack_fn(images)
            defended = defense_fn(attacked)

            preds = model(defended).argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# =====================================================
# ADAPTIVE DEFENSE
# =====================================================
def adaptive_defense(x, attack_name):

    if attack_name == "gaussian_noise":
        return denoise_images(x, method="smoothing")

    elif attack_name == "bit_depth":
        return feature_squeezing(x, bits=6)

    elif attack_name == "blur":
        return denoise_images(x, method="sharpen")

    elif attack_name == "resize":
        return resize_smoothing(x)

    elif attack_name == "color":
        return color_normalization(x)

    return x


# =====================================================
# STAGE 2
# =====================================================
def run_stage2():

    train_loader, test_loader = get_dataloaders("cifar10", batch_size=64)

    # =====================================================
    # CLEAN TRAINING
    # =====================================================
    print("\n========== CLEAN TRAINING ==========")

    clean_model = SimpleCNN(num_classes=10).to(device)
    clean_model = train(clean_model, train_loader)

    clean_acc = clean_accuracy(clean_model, test_loader, device)
    print("Clean Accuracy:", clean_acc)

    # =====================================================
    # ATTACKS
    # =====================================================
    print("\n========== PREPROCESSING ATTACKS ==========")

    attacks = {
        "gaussian_noise": lambda x: add_gaussian_noise(x, std=0.03),
        "bit_depth": lambda x: bit_depth_reduction_attack(x, bits=6),
        "blur": lambda x: gaussian_blur_attack(x, kernel_size=5),
        "resize": lambda x: resize_attack(x, scale=0.9),
        "color": lambda x: color_channel_attack(x, strength=0.05),
    }

    attack_results = {}

    for name, attack_fn in attacks.items():
        acc = evaluate_attack(clean_model, test_loader, attack_fn)
        asr = 1 - acc

        attack_results[name] = (acc, asr)

        print(f"\n{name}")
        print(f"Attack Accuracy: {acc:.4f}")
        print(f"ASR: {asr:.4f}")

    # =====================================================
    # DEFENSES
    # =====================================================
    print("\n========== PREPROCESSING DEFENSE ==========")

    defense_results = {}
    def bit_depth_defense(x):
        return feature_squeezing(x, bits=5)

    defenses = {
        "gaussian_noise": denoise_images,
        "bit_depth": bit_depth_defense,
        "blur": lambda x: denoise_images(x, method="sharpen"),
        "resize": resize_smoothing,
        "color": color_normalization,
    }

    for name in attacks.keys():

        attack_fn = attacks[name]
        defense_fn = defenses[name]

        attacked_acc, asr = attack_results[name]

        # 🔹 Single defense
        single_acc = evaluate_defense(
            clean_model,
            test_loader,
            attack_fn,
            defense_fn
        )

        # 🔹 Adaptive defense
        adaptive_acc = evaluate_defense(
            clean_model,
            test_loader,
            attack_fn,
            lambda x: adaptive_defense(x, name)
        )

        gain = defense_gain(attacked_acc, single_acc)
        adaptive_gain = defense_gain(attacked_acc, adaptive_acc)

        defense_results[name] = {
            "single": single_acc,
            "adaptive": adaptive_acc,
            "gain": gain,
            "adaptive_gain": adaptive_gain
        }

    # =====================================================
    # FINAL RESULTS
    # =====================================================
    print("\n========== FINAL RESULTS ==========")

    for name in attacks.keys():

        attacked_acc, asr = attack_results[name]
        result = defense_results[name]

        print(f"\n{name}")
        print(f"Attack Acc: {attacked_acc:.4f}")
        print(f"Single Defense Acc: {result['single']:.4f}")
        print(f"Adaptive Defense Acc: {result['adaptive']:.4f}")
        print(f"ASR: {asr:.4f}")
        print(f"Gain (Single): {result['gain']:.4f}")
        print(f"Gain (Adaptive): {result['adaptive_gain']:.4f}")


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    run_stage2()