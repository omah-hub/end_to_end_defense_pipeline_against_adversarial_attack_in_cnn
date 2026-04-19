import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


# ==========================================================
# 1️⃣ Classification Metrics
# ==========================================================

def accuracy(model, dataloader, device):
    """
    Standard classification accuracy.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def clean_accuracy(model, dataloader, device):
    """
    Accuracy on clean (unpoisoned) test data.
    """
    return accuracy(model, dataloader, device)


def robust_accuracy(model, dataloader, device):
    """
    Accuracy under attack (poisoned or adversarial test set).
    """
    return accuracy(model, dataloader, device)


# ==========================================================
# 2️⃣ Attack Metrics
# ==========================================================

def attack_success_rate(model, dataloader, target_class, device):
    """
    For targeted attacks (e.g., backdoor).
    Measures how often inputs are classified as target class.
    """
    model.eval()
    total = 0
    success = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            success += (preds == target_class).sum().item()
            total += preds.size(0)

    return success / total


def misclassification_rate(model, dataloader, device):
    """
    General misclassification rate.
    """
    model.eval()
    total = 0
    incorrect = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            incorrect += (preds != labels).sum().item()
            total += labels.size(0)

    return incorrect / total


# ==========================================================
# 3️⃣ Defense Evaluation Metrics
# ==========================================================

def defense_gain(poisoned_acc, defended_acc):
    """
    Improvement after applying defense.
    """
    return defended_acc - poisoned_acc


def clean_accuracy_drop(clean_acc, defended_acc):
    """
    Measures how much defense affects normal performance.
    """
    return clean_acc - defended_acc


# ==========================================================
# 4️⃣ Detector Metrics
# ==========================================================

def detection_rate(detector_preds):
    """
    True Positive Rate (Recall) for poisoned samples.
    detector_preds: tensor of 0/1 (1 = detected as poison)
    """
    return detector_preds.sum().item() / detector_preds.size(0)


def false_positive_rate(y_true, y_pred):
    """
    Standard false positive rate for binary detector:
    FPR = FP / (FP + TN)
    y_true: 0/1 labels
    y_pred: 0/1 predictions
    """
    # Convert to numpy
    y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)
    y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else np.array(y_pred)

    # Ensure binary
    y_true = (y_true > 0).astype(int)
    y_pred = (y_pred > 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp = cm[0]
    fn, tp = cm[1]

    fpr = fp / (fp + tn + 1e-8)
    return fpr


def precision(detector_preds, true_labels):
    """
    Precision for detector.
    """
    tp = ((detector_preds == 1) & (true_labels == 1)).sum().item()
    fp = ((detector_preds == 1) & (true_labels == 0)).sum().item()

    return tp / (tp + fp + 1e-8)


def recall(detector_preds, true_labels):
    """
    Recall (same as detection rate).
    """
    tp = ((detector_preds == 1) & (true_labels == 1)).sum().item()
    fn = ((detector_preds == 0) & (true_labels == 1)).sum().item()

    return tp / (tp + fn + 1e-8)


# ==========================================================
# 5️⃣ Advanced Metrics (Optional but Strong for Research)
# ==========================================================

def confusion_matrix_metrics(y_true, y_pred):
    """
    Returns confusion matrix values.
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def roc_auc_score_metric(y_true, y_scores):
    """
    ROC-AUC for binary detector.
    y_true: 0/1 labels
    y_scores: continuous probability scores (sigmoid outputs)
    """
    # Convert to numpy
    y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)
    y_scores = y_scores.cpu().numpy() if torch.is_tensor(y_scores) else np.array(y_scores)

    # Ensure 1D
    y_true = y_true.ravel()
    y_scores = y_scores.ravel()

    return roc_auc_score(y_true, y_scores)