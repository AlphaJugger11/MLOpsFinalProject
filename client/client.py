# client/client.py
import argparse
import flwr as fl
import torch
import os
from pathlib import Path
from client.data_prep import load_client_csv
from shared.model_utils import get_parameters_numpy, set_parameters_from_numpy, load_model_torch
from shared.model import get_model
from prometheus_client import Gauge, start_http_server

local_acc_gauge = Gauge("client_local_accuracy", "Local validation accuracy", ["client_id"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parameters(model):
    return get_parameters_numpy(model)

def set_parameters(model, parameters):
    set_parameters_from_numpy(model, parameters)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, client_id):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.client_id = client_id
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def get_parameters(self):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.train()
        epochs = int(config.get("local_epochs", 1))
        for _ in range(epochs):
            for Xb, yb in self.train_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                logits = self.model(Xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
        loss = 0.0; correct = 0; total = 0
        with torch.no_grad():
            for Xb, yb in self.val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = self.model(Xb)
                loss += self.criterion(logits, yb).item() * Xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += Xb.size(0)
        acc = float(correct/total) if total>0 else 0.0
        local_acc_gauge.labels(client_id=self.client_id).set(acc)
        return float(loss/total) if total>0 else 0.0, total, {'accuracy': acc}

def start_client(fl_server, client_id, data_path):
    start_http_server(9100)  # expose prometheus metrics
    train_loader, val_loader, in_dim, num_classes, _ = load_client_csv(data_path)
    model = get_model(in_dim=in_dim, num_classes=num_classes)
    # try load latest global model from ../models
    models_dir = Path.cwd().parents[1] / 'models'
    if models_dir.exists():
        model_files = sorted(models_dir.glob('global_round_*.pth'))
        if model_files:
            latest = model_files[-1]
            try:
                load_model_torch(model, str(latest), map_location=DEVICE)
                print(f'[client:{client_id}] loaded global model {latest}')
            except Exception as e:
                print('[client] load model error:', e)
    client = FlowerClient(model, train_loader, val_loader, client_id)
    fl.client.start_numpy_client(server_address=fl_server, client=client)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--fl_server', default='127.0.0.1:8080')
    args = parser.parse_args()
    start_client(args.fl_server, args.client_id, args.data)
