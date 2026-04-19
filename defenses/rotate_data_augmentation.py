import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.ToTensor()
])