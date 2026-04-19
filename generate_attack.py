import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models.simple_cnn import SimpleCNN
from attacks.cw import cw_attack

device = "cuda" if torch.cuda.is_available() else "cpu"

# CIFAR10 labels
CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# Load trained model
model = SimpleCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("saved_models/simple_cnn_robust_pgd.pth", map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

# Load image
image_path = "cifar_sample2.png"
image = Image.open(image_path).convert("RGB")

image_tensor = transform(image).unsqueeze(0).to(device)

# True label (example: cat = 3)
label = torch.tensor([3]).to(device)

# Generate adversarial image
epsilon = 0.03
adv_image = cw_attack(model, image_tensor, label)

# Save adversarial image
vutils.save_image(adv_image, "pgd_adversarial2.png")

print("FGSM adversarial image saved as fgsm_adversarial.png")