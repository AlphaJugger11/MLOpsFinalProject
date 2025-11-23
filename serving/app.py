# serving/app.py
from fastapi import FastAPI
import torch
from pydantic import BaseModel
from shared.model import get_model
from shared.model_utils import load_model_torch
import os

app = FastAPI()
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/global_round_0.pth')
model = None

class PredictRequest(BaseModel):
    features: list

@app.on_event('startup')
def load_model():
    global model
    in_dim = int(os.environ.get('IN_DIM', 561))
    num_classes = int(os.environ.get('NUM_CLASSES', 6))
    model = get_model('medium', in_dim=in_dim, num_classes=num_classes)
    try:
        load_model_torch(model, MODEL_PATH, map_location='cpu')
        model.eval()
        print('[serving] loaded model', MODEL_PATH)
    except Exception as e:
        print('[serving] could not load model:', e)

@app.post('/predict')
def predict(req: PredictRequest):
    import torch
    x = torch.tensor([req.features], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        pred = int(logits.argmax(dim=1).item())
    return {'prediction': pred}
