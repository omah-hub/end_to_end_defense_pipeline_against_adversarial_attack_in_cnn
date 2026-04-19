# deployment/defense_system.py

import torch
from models.simple_cnn import SimpleCNN
from defenses.preprocessing_defense import denoise_images
from defenses.feature_squeezing import feature_squeezing
from defenses.detector import Detector

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]
class DefenseSystem:
    def __init__(self, model_path, detector_path, device=None):

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load robust model
        self.model = SimpleCNN(num_classes=10).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Load detector
        self.detector = Detector(input_dim=10).to(self.device)
        self.detector.load_state_dict(torch.load(detector_path, map_location=self.device))
        self.detector.eval()

        print("✅ Defense system deployed successfully.")

    def run_inference(self, image_tensor):

      image_tensor = image_tensor.to(self.device)

      # 1️⃣ Initial prediction
      with torch.no_grad():
          logits = self.model(image_tensor)
          probs = torch.softmax(logits, dim=1)
          confidence, prediction = torch.max(probs, dim=1)

      # 2️⃣ Detection
      with torch.no_grad():
          det_score = self.detector(logits)
          score = det_score.item()

      is_adversarial = (score > 0.5)

      # 3️⃣ If adversarial → apply defense + re-evaluate
      if is_adversarial:

          # Apply defenses
          defended = denoise_images(image_tensor, method="smoothing")
          defended = feature_squeezing(defended, bits=5)

          with torch.no_grad():
              logits_def = self.model(defended)
              probs_def = torch.softmax(logits_def, dim=1)
              confidence_def, prediction_def = torch.max(probs_def, dim=1)

          confidence = confidence_def.item()
          prediction = prediction_def.item()

          # 🔥 SECOND DETECTION AFTER DEFENSE (VERY IMPORTANT)
          with torch.no_grad():
              new_score = self.detector(logits_def).item()

          # 🚨 Final rejection rule
          if new_score > 0.5:
              return {
                  "status": "Rejected",
                  "reason": "Adversarial input persists after defense",
                  "detector_score": new_score
              }

      else:
          prediction = prediction.item()
          confidence = confidence.item()

      # 4️⃣ Confidence check
      if confidence < 0.4:
          return {
              "status": "Rejected",
              "reason": "Low confidence prediction",
              "confidence": confidence
          }

      label = CIFAR10_CLASSES[prediction]

      return {
          "status": "Accepted",
          "prediction_index": prediction,
          "prediction_label": label,
          "confidence": confidence,
          "adversarial_detected": bool(is_adversarial),
          "detector_score": score
      }