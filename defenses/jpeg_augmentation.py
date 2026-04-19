from PIL import Image
import io

def jpeg_compress(img, quality=30):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)