import torchvision
import torchvision.transforms as transforms
from PIL import Image

# Load CIFAR10 test set
transform = transforms.ToTensor()
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Get one image
image, label = testset[3]

# Convert tensor to PIL image
to_pil = transforms.ToPILImage()
pil_image = to_pil(image)

# Save it
pil_image.save("cifar_sample3.png")

print("Saved cifar_sample.png")