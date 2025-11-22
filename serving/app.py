# serving/app.py
from fastapi import FastAPI
import torch
from pydantic import BaseModel
from pathlib import Path
from shared.model import get_model

app = FastAPI()
MODEL_PATH = Path.cwd().parent / "models" / "global_round_0.pth"
model = None

class PredictRequest(BaseModel):
    features: list

@app.on_event('startup')
def load_model():
    global model
    in_dim = int(__import__('os').environ.get('IN_DIM', '561'))
    num_classes = int(__import__('os').environ.get('NUM_CLASSES', '6'))
    model = get_model(in_dim=in_dim, num_classes=num_classes)
    try:
        if MODEL_PATH.exists():
            model.load_state_dict(torch.load(str(MODEL_PATH), map_location='cpu'))
            model.eval()
            print('[serving] loaded model', MODEL_PATH)
        else:
            print('[serving] no model found at', MODEL_PATH)
    except Exception as e:
        print('[serving] error loading model:', e)

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
