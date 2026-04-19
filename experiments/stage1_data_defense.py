import os
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.nn as nn
from torch.optim import Adam

import random
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from data.data_loader import get_dataloaders
from models.simple_cnn import SimpleCNN
from attacks.backdoor import backdoor_poison, add_trigger
from defenses.data_sanitization import filter_high_loss_samples
from attacks.adaptive_attack import adaptive_combined_attack
from attacks.poison import label_flip
from defenses.backdoor_defense import fine_pruning
from defenses.strip_defense import strip_defense
from defenses.adaptive_defense import adaptive_defense_training
from defenses.spectral_signature import spectral_signature_filter
from evaluation.metrics import (
    clean_accuracy,
    attack_success_rate,
    defense_gain,
    clean_accuracy_drop
)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# TRAIN FUNCTION
# =====================================================

def train(model, dataloader, epochs=10):
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


def run_stage1(poison_rate=0.1, target_class=0):

    train_loader, test_loader = get_dataloaders("cifar10", batch_size=64)

    # =====================================================
    # 1️⃣ CLEAN TRAINING
    # =====================================================

    print("\n========== CLEAN TRAINING ==========")

    clean_model = SimpleCNN(num_classes=10).to(device)
    clean_model = train(clean_model, train_loader)

    clean_acc = clean_accuracy(clean_model, test_loader, device)
    print("Clean Accuracy:", clean_acc)

    # =====================================================
    # 2️⃣ LABEL FLIP TRAINING
    # =====================================================

    print("\n========== LABEL FLIP ATTACK ==========")

    # flipped_images = []
    # flipped_labels = []

    # for images, labels in train_loader:

    #     labels = label_flip(
    #         labels,
    #         poison_rate=poison_rate,
    #         num_classes=10
    #     )

    #     flipped_images.append(images)
    #     flipped_labels.append(labels)

    # flipped_images = torch.cat(flipped_images)
    # flipped_labels = torch.cat(flipped_labels)

    # label_flip_dataset = TensorDataset(flipped_images, flipped_labels)
    # label_flip_loader = DataLoader(label_flip_dataset, batch_size=64, shuffle=True)

    # label_flip_model = SimpleCNN(num_classes=10).to(device)
    # label_flip_model = train(label_flip_model, label_flip_loader)

    # lf_acc = clean_accuracy(label_flip_model, test_loader, device)
    # print("Label Flip Accuracy:", lf_acc)

    # =====================================================
    # 3️⃣ BACKDOOR ATTACK
    # =====================================================

    print("\n========== BACKDOOR ATTACK ==========")

    print("\n========== BACKDOOR ATTACK ==========")

    poisoned_images = []
    poisoned_labels = []

    for images, labels in train_loader:

        images = images.clone()
        labels = labels.clone()

        mask = torch.rand(labels.size(0)) < poison_rate

        # Apply trigger only to selected samples
        images[mask] = add_trigger(images[mask], trigger_size=4)
        labels[mask] = target_class

        poisoned_images.append(images)
        poisoned_labels.append(labels)

    poisoned_images = torch.cat(poisoned_images, dim=0)
    poisoned_labels = torch.cat(poisoned_labels, dim=0)

    # Create dataset and loader
    backdoor_dataset = TensorDataset(poisoned_images, poisoned_labels)

    backdoor_loader = DataLoader(
        backdoor_dataset,
        batch_size=64,
        shuffle=True
    )

    # Train backdoor model
    backdoor_model = SimpleCNN(num_classes=10).to(device)
    backdoor_model = train(backdoor_model, backdoor_loader)

    # Evaluate clean accuracy
    bd_acc = clean_accuracy(backdoor_model, test_loader, device)
    print("Backdoor Accuracy:", bd_acc)

    # Attack Success Rate
    test_images = []
    for x, _ in test_loader:
        test_images.append(x)
    test_images = torch.cat(test_images, dim=0)
    triggered_images = add_trigger(test_images, trigger_size=4)

    triggered_labels = torch.full(
        (triggered_images.size(0),),
        target_class
    )

    triggered_loader = DataLoader(
        TensorDataset(triggered_images, triggered_labels),
        batch_size=64
    )

    asr = attack_success_rate(
        backdoor_model,
        triggered_loader,
        target_class,
        device
    )

    print("Backdoor ASR:", asr)

    # =====================================================
    # 4️⃣ ADAPTIVE ATTACK
    # =====================================================

    print("\n========== ADAPTIVE ATTACK ==========")

    # adaptive_images = []
    # adaptive_labels = []

    # for images, labels in train_loader:

    #     images, labels = adaptive_combined_attack(
    #         images,
    #         labels,
    #         target_label=target_class,
    #         poison_rate=poison_rate
    #     )

    # adaptive_images.append(images)
    # adaptive_labels.append(labels)

    # adaptive_images = torch.cat(adaptive_images)
    # adaptive_labels = torch.cat(adaptive_labels)


    # adaptive_loader = DataLoader(
    #     TensorDataset(adaptive_images, adaptive_labels),
    #     batch_size=64,
    #     shuffle=True
    # )

    # adaptive_model = SimpleCNN(num_classes=10).to(device)
    # adaptive_model = train(adaptive_model, adaptive_loader)

    # adaptive_acc = clean_accuracy(adaptive_model, test_loader, device)
    # print("Adaptive Accuracy:", adaptive_acc)

    # adaptive_asr = attack_success_rate(
    #     adaptive_model,
    #     triggered_loader,
    #     target_class,
    #     device
    # )

    # print("Adaptive ASR:", adaptive_asr)

# =====================================================
# 5️⃣ DEFENSE
# =====================================================

    print("\n========== LABEL FLIP DEFENSE ==========")

    # # ----- Label Flip Defense -----
    # filtered_loader = filter_high_loss_samples(
    #     label_flip_model,
    #     label_flip_loader,
    #     device
    # )

    # label_defended_model = SimpleCNN(num_classes=10).to(device)
    # label_defended_model = train(label_defended_model, filtered_loader)

    # label_defended_acc = clean_accuracy(
    #     label_defended_model, test_loader, device
    # )

    # print("Label Defended Accuracy:", label_defended_acc)


    # =====================================================

    print("\n========== BACKDOOR DEFENSE (STRIP) ==========")

    # Run STRIP detection on the attacked model
    print("\n========== BACKDOOR DEFENSE (SPECTRAL SIGNATURE) ==========")

    print("\n========== BACKDOOR DEFENSE (SPECTRAL SIGNATURE) ==========")

    clean_loader_ss = spectral_signature_filter(
        backdoor_model,
        backdoor_loader,
        device,
        remove_ratio=0.1
    )

    ss_model = SimpleCNN(num_classes=10).to(device)
    ss_model = train(ss_model, clean_loader_ss)

    ss_acc = clean_accuracy(ss_model, test_loader, device)
    ss_asr = attack_success_rate(ss_model, triggered_loader, target_class, device)

    print("Spectral Signature Accuracy:", ss_acc)
    print("Spectral Signature ASR:", ss_asr)

    # =====================
    # OPTIONAL: FINE PRUNING
    # =====================
    print("\n========== FINE PRUNING ==========")

    pruned_model = fine_pruning(ss_model, clean_loader_ss, device)

    pruned_acc = clean_accuracy(pruned_model, test_loader, device)
    pruned_asr = attack_success_rate(pruned_model, triggered_loader, target_class, device)

    print("Pruned Accuracy:", pruned_acc)
    print("Pruned ASR:", pruned_asr)

    # =====================
    # STRIP (CORRECT USAGE)
    # =====================
    print("\n========== STRIP (DETECTION ONLY) ==========")

    _, strip_rate = strip_defense(
        backdoor_model,
        triggered_loader,
        train_loader,
        device
    )

    print("STRIP Detection Rate:", strip_rate)


    # =====================================================

    print("\n========== ADAPTIVE DEFENSE ==========")

    # adaptive_defended_model = adaptive_defense_training(
    #     adaptive_model,
    #     adaptive_loader,
    #     device
    # )

    # adaptive_defended_acc = clean_accuracy(
    #     adaptive_defended_model,
    #     test_loader,
    #     device
    # )

    # adaptive_defended_asr = attack_success_rate(
    #     adaptive_defended_model,
    #     triggered_loader,
    #     target_class,
    #     device
    # )

    # print("Adaptive Defended Accuracy:", adaptive_defended_acc)
    # print("Adaptive Defended ASR:", adaptive_defended_asr)


    # =====================================================
    # 6️⃣ FINAL METRICS
    # =====================================================

    print("\n========== FINAL RESULTS ==========")

    print("Clean Accuracy:", clean_acc)
    # print("Label Flip Accuracy:", lf_acc)
    print("Backdoor Accuracy:", bd_acc)
    # print("Adaptive Accuracy:", adaptive_acc)

    print("Backdoor ASR:", asr)
    # print("Adaptive ASR:", adaptive_asr)

    print("\n--- After Defense ---")

    # print("Label Defended Accuracy:", label_defended_acc)
    print("Spectral Signature Accuracy:", ss_acc)
    print("Spectral Signature ASR:", ss_asr)
    print("Pruned Accuracy:", pruned_acc)
    print("Pruned ASR:", pruned_asr)
    # print("Adaptive Defended Accuracy:", adaptive_defended_acc)
    # print("Adaptive Defended ASR:", adaptive_defended_asr)

    # =====================================================
    # Improvement Metrics
    # =====================================================

    print("\n--- Defense Improvements ---")

    # print("Backdoor Reduction:", asr - backdoor_defended_asr)
    # print("Adaptive Reduction:", adaptive_asr - adaptive_defended_asr)

    # print("Clean Accuracy Drop:",
    #       clean_acc - backdoor_defended_acc)


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    run_stage1()
