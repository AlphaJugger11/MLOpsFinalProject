# serving/app.py
from fastapi import FastAPI
import torch
from pydantic import BaseModel
from pathlib import Path
from shared.model import get_model

app = FastAPI()
MODELS_DIR = Path("/app/models")  # mounted volume
model = None

class PredictRequest(BaseModel):
    features: list

def get_latest_model_path(models_dir: Path):
    all_models = list(models_dir.glob("global_round_*.pth"))
    if not all_models:
        return None
    # Sort by round number extracted from filename
    all_models.sort(key=lambda p: int(p.stem.split("_")[-1]), reverse=True)
    return all_models[0]

@app.on_event('startup')
def load_model():
    global model
    in_dim = int(__import__('os').environ.get('IN_DIM', '561'))
    num_classes = int(__import__('os').environ.get('NUM_CLASSES', '6'))
    model = get_model(in_dim=in_dim, num_classes=num_classes)

    model_path = get_latest_model_path(MODELS_DIR)
    if model_path is None:
        print("[serving] WARNING: No FL model found in", MODELS_DIR)
        return

    try:
        model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        model.eval()
        print(f"[serving] Loaded latest model: {model_path}")
    except Exception as e:
        print("[serving] Error loading model:", e)

@app.post('/predict')
def predict(req: PredictRequest):
    import torch
    if model is None:
        return {'error': 'no model available'}
    x = torch.tensor([req.features], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        pred = int(logits.argmax(dim=1).item())
    return {'prediction': pred}

@app.get('/health')
def health():
    return {'status':'ok'}
