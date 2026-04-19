import os
import sys
import torch
from tqdm import tqdm

# ===== Project Root =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# ===== Imports =====
from models.simple_cnn import SimpleCNN
from data.data_loader import get_dataloaders

# Classical adversarial attacks
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.deepfool import deepfool_attack
from attacks.cw import cw_attack

# Preprocessing-style attacks
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
    accuracy, clean_accuracy, robust_accuracy,
    attack_success_rate, misclassification_rate,
    defense_gain, clean_accuracy_drop,
    detection_rate, false_positive_rate, precision, recall,
    confusion_matrix_metrics, roc_auc_score_metric
)

# ===== Device =====
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Load Models =====
robust_model_path = os.path.join(PROJECT_ROOT, "saved_models", "simple_cnn_robust_multi_attack.pth")
model = SimpleCNN(num_classes=10).to(device)
model.load_state_dict(torch.load(robust_model_path, map_location=device))
model.eval()

# ===== Load Detector (Optional) =====
detector_path = os.path.join(PROJECT_ROOT, "saved_models", "detector_cifar10.pth")
if os.path.exists(detector_path):
    from defenses.detector import Detector
    detector = Detector(input_dim=10).to(device)
    detector.load_state_dict(torch.load(detector_path, map_location=device))
    detector.eval()
    print("Detector loaded successfully!")
else:
    detector = None
    print("No detector found. Detection will be skipped.")

# ===== Load Data =====
train_loader, test_loader = get_dataloaders("cifar10", batch_size=64)

# ===== Preprocessing Defenses and Attacks =====
preprocessing_defenses = {
    "denoise": denoise_images,
    "feature_squeezing": feature_squeezing,
    "resize_smoothing": resize_smoothing,
    "color_normalization": color_normalization,
}

preprocessing_attacks = {
    "gaussian_noise": add_gaussian_noise,
    "bit_depth": bit_depth_reduction_attack,
    "blur": gaussian_blur_attack,
    "resize": resize_attack,
    "color": color_channel_attack,
}

# Classical adversarial attacks
adversarial_attacks = ["FGSM", "PGD", "Deepfool", "CW"]

# ===== Helper: Classical Adversarial Evaluation =====
def evaluate_adversarial(model, attack_name, test_loader, detector=None, device="cpu", epsilon=0.03):
    correct = 0
    total = 0
    detected_total = 0

    for images, labels in tqdm(test_loader, desc=f"Evaluating {attack_name}"):
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial examples
        if attack_name.lower() == "fgsm":
            adv_images = fgsm_attack(model, images, labels, epsilon=epsilon)
        elif attack_name.lower() == "pgd":
            adv_images = pgd_attack(model, images, labels, epsilon=epsilon, alpha=0.007, iters=10)
        elif attack_name.lower() == "deepfool":
            adv_images = deepfool_attack(model, images, labels=labels)
        elif attack_name.lower() == "cw":
            adv_images = cw_attack(model, images, labels)
        else:
            raise ValueError(f"Unknown attack: {attack_name}")

        # Apply default preprocessing defense
        adv_images_defended = denoise_images(adv_images).to(device)

        # Model predictions
        with torch.no_grad():
            logits = model(adv_images_defended)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Detector evaluation
        if detector is not None:
            with torch.no_grad():
                det_preds = detector(logits)
                detected_total += (det_preds > 0.5).int().sum().item()

    acc = correct / total
    detection_rate = detected_total / total if detector is not None else None
    return acc, detection_rate

# ===== Helper: Preprocessing Attack Evaluation =====
def evaluate_preprocessing_attack(model, attack_fn, test_loader, defense_fn=None, detector=None, device="cpu", attack_params={}):
    correct = 0
    total = 0
    detected_total = 0

    for images, labels in tqdm(test_loader, desc=f"Evaluating {attack_fn.__name__}"):
        images, labels = images.to(device), labels.to(device)

        # Generate preprocessing attack
        adv_images = attack_fn(images, **attack_params).to(device)

        # Apply preprocessing defense
        if defense_fn is not None:
            adv_images = defense_fn(adv_images).to(device)

        # Model predictions
        with torch.no_grad():
            logits = model(adv_images)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Detector evaluation
        if detector is not None:
            with torch.no_grad():
                det_preds = detector(logits)
                detected_total += (det_preds > 0.5).int().sum().item()

    acc = correct / total
    detection_rate = detected_total / total if detector is not None else None
    return acc, detection_rate

# ===== Run Full Evaluation =====
results = {}

# 1️⃣ Classical Adversarial Attacks
print("\n=== Classical Adversarial Attacks ===")
for atk in adversarial_attacks:
    acc, det_rate = evaluate_adversarial(model, atk, test_loader, detector=detector, device=device)
    results[atk] = {"accuracy": acc, "detection_rate": det_rate}
    print(f"{atk} - Accuracy: {acc:.4f}", end="")
    if det_rate is not None:
        print(f", Detection rate: {det_rate:.4f}")
    else:
        print("")

# 2️⃣ Preprocessing Attacks × Defenses
print("\n=== Preprocessing Attacks × Defenses ===")
for attack_name, attack_fn in preprocessing_attacks.items():
    results[attack_name] = {}
    for defense_name, defense_fn in preprocessing_defenses.items():
        acc, det_rate = evaluate_preprocessing_attack(
            model,
            attack_fn,
            test_loader,
            defense_fn=defense_fn,
            detector=detector,
            device=device
        )
        results[attack_name][defense_name] = {"accuracy": acc, "detection_rate": det_rate}
        print(f"{attack_name} -> {defense_name} : Accuracy = {acc:.4f}", end="")
        if det_rate is not None:
            print(f", Detection rate: {det_rate:.4f}")
        else:
            print("")

    # Attack with no defense
    acc, det_rate = evaluate_preprocessing_attack(
        model,
        attack_fn,
        test_loader,
        defense_fn=None,
        detector=detector,
        device=device
    )
    results[attack_name]["no_defense"] = {"accuracy": acc, "detection_rate": det_rate}
    print(f"{attack_name} -> no defense : Accuracy = {acc:.4f}", end="")
    if det_rate is not None:
        print(f", Detection rate: {det_rate:.4f}")
    else:
        print("")

# 3️⃣ Clean Accuracy for reference
clean_acc = clean_accuracy(model, test_loader, device)
print(f"\nClean Accuracy (no attack, no defense): {clean_acc:.4f}")