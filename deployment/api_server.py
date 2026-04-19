from fastapi.security import APIKeyHeader
from fastapi.openapi.utils import get_openapi
import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request, Header, HTTPException,Security
from dotenv import load_dotenv
from PIL import Image
import torch
from torchvision import transforms
from deployment.defense_system import DefenseSystem

# rate limiter
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

load_dotenv(dotenv_path=".env")
app = FastAPI()
API_KEY = os.getenv("API_KEY")
# ✅ Define API Key header schema (this makes Swagger show "Authorize")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
# limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

device = "cuda" if torch.cuda.is_available() else "cpu"

system = DefenseSystem(
    model_path="saved_models/simple_cnn_robust_multi_attack.pth",
    detector_path="saved_models/detector_cifar10.pth",
    device=device
)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

@app.post("/predict")
@limiter.limit("5/minute")  # 👈 query limit
async def predict(request: Request, file: UploadFile = File(...),api_key: str = Security(api_key_header)):
    verify_api_key(api_key)
    image = Image.open(file.file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    result = system.run_inference(image_tensor)

    return result

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Defense API",
        version="1.0.0",
        description="Adversarial Defense System API",
        routes=app.routes,
    )

    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "x-api-key"
        }
    }

    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"ApiKeyAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
