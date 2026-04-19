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
from defenses.detector import Detector

# Metrics
from evaluation.metrics import (
    precision,
    recall,
    false_positive_rate,
    confusion_matrix_metrics,
    roc_auc_score_metric
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Load robust model
# ---------------------------
def load_model():
    model = SimpleCNN(num_classes=10).to(device)
    path = os.path.join(PROJECT_ROOT, "saved_models", "simple_cnn_robust_multi_attack.pth")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# ---------------------------
# Evaluate accuracy and metrics under attack
# ---------------------------
def evaluate_attack(model, attack_name="fgsm", epsilon=0.03, alpha=0.007, detector=None):
    train_loader, test_loader = get_dataloaders("cifar10", batch_size=64)
    
    all_labels = []
    all_preds = []

    detector_labels = []
    detector_probs = []

    # -----------------------------
    # Go through test data
    # -----------------------------
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial examples
        if attack_name == "fgsm":
            adv_images = fgsm_attack(model, images, labels, epsilon=epsilon)
        elif attack_name == "pgd":
            adv_images = pgd_attack(model, images, labels, epsilon=epsilon, alpha=alpha, iters=10)
        elif attack_name == "deepfool":
            adv_images = deepfool_attack(model, images, labels)
        elif attack_name == "cw":
            adv_images = cw_attack(model, images, labels)
        else:
            raise ValueError("Unknown attack type")

        # -----------------------------
        # Model predictions
        # -----------------------------
        with torch.no_grad():
            outputs = model(adv_images)
            preds = outputs.argmax(1)

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())

        # -----------------------------
        # Detector predictions
        # -----------------------------
        if detector:
            with torch.no_grad():
                # Detector on clean
                clean_outputs = model(images)
                clean_probs = detector(clean_outputs).squeeze(1)
                detector_probs.append(clean_probs.cpu())
                detector_labels.append(torch.zeros_like(clean_probs.cpu()))  # clean = 0

                # Detector on adversarial
                adv_probs = detector(outputs).squeeze(1)
                detector_probs.append(adv_probs.cpu())
                detector_labels.append(torch.ones_like(adv_probs.cpu()))   # adv = 1

    # -----------------------------
    # Flatten batches
    # -----------------------------
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    
    if detector:
        all_detector_probs = torch.cat(detector_probs)
        all_detector_labels = torch.cat(detector_labels)

    # -----------------------------
    # Compute metrics
    # -----------------------------
    acc = (all_preds == all_labels).float().mean().item()
    prec = precision(all_labels, all_preds)
    rec = recall(all_labels, all_preds)
    cm = confusion_matrix_metrics(all_labels, all_preds)

    if detector:
    # Binary detector predictions
      detection_preds = (all_detector_probs > 0.5).float()

      # FPR only on clean samples
      y_true_clean = all_detector_labels[all_detector_labels==0]
      y_pred_clean = detection_preds[all_detector_labels==0]
      fpr = false_positive_rate(y_true_clean, y_pred_clean)

      # ROC-AUC on all samples (both clean + adv)
      roc_auc = roc_auc_score_metric(all_detector_labels, all_detector_probs)

      # Detection rate (TPR) on adversarial samples
      y_true_adv = all_detector_labels[all_detector_labels==1]
      y_pred_adv = detection_preds[all_detector_labels==1]
      detection_rate = recall(y_pred_adv, y_true_adv)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "false_positive_rate": fpr,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "detection_rate": detection_rate
    }

# ---------------------------
# Main evaluation
# ---------------------------
if __name__ == "__main__":
    # Optionally load detector
    detector_path = os.path.join(PROJECT_ROOT, "saved_models", "detector_cifar10.pth")
    detector = Detector(input_dim=10).to(device)
    if os.path.exists(detector_path):
        detector.load_state_dict(torch.load(detector_path, map_location=device))
        detector.eval()
    else:
        detector = None

    robust_model = load_model()

    attacks = ["fgsm", "pgd", "deepfool", "cw"]
    results = {}

    for atk in attacks:
        metrics = evaluate_attack(robust_model, attack_name=atk, detector=detector)
        results[atk] = metrics

        print(f"\n{atk.upper()} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"FPR: {metrics['false_positive_rate']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Detection Rate: {metrics['detection_rate'] if detector else 'N/A'}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

    print("\n================ Summary =================")
    print("Attack | Accuracy | Precision | Recall | FPR | ROC-AUC | Detection")
    for atk, m in results.items():
        det_str = f"{m['detection_rate']:.4f}" if detector else "N/A"
        print(f"{atk.upper():<7} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['false_positive_rate']:.4f} | {m['roc_auc']:.4f} | {det_str}")