import torch
import torchvision.transforms.functional as TF
from PIL import Image
import io

def jpeg_compression_attack(images, quality=30):
    compressed_images = []

    for img in images:
        pil_img = TF.to_pil_image(img.cpu())

        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        compressed = Image.open(buffer)
        compressed = TF.to_tensor(compressed)

        compressed_images.append(compressed)

    return torch.stack(compressed_images).to(images.device)