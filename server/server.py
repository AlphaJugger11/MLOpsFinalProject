# server/server.py
import threading
import os
import socket
from pathlib import Path
from flask import Flask, request, jsonify
import mlflow
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays
from shared.model_utils import set_parameters_from_numpy, save_model_torch, set_state_dict_from_numpy_by_keys
from server.model import get_model
from flwr.server import ServerConfig
app = Flask(__name__)
MODELS_DIR = Path.cwd() / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FL_PORT_DEFAULT = 8080

def find_free_port(start_port=8080):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                port += 1

# FL_PORT = find_free_port(FL_PORT_DEFAULT)
FL_PORT = int(os.environ.get("FL_PORT", 8080))

print(f"[server] Flower server will run on port {FL_PORT}")

@app.route('/list_clients', methods=['GET'])
def list_clients():
    return jsonify({'info': 'no profiling in simplified flow'})

@app.route('/start_rounds', methods=['POST'])
def start_rounds():
    data = request.json or {}
    rounds = int(data.get('rounds', 1))
    t = threading.Thread(target=lambda: run_flower_server(rounds), daemon=True)
    t.start()
    return jsonify({'status': 'started', 'rounds': rounds, 'fl_port': FL_PORT})

@app.route('/get_fl_port', methods=['GET'])
def get_fl_port():
    return jsonify({'fl_port': FL_PORT})

class SaveModelFedAvg(FedAvg):
    def __init__(self, *args, base_in_dim=561, base_num_classes=6, **kwargs):
        super().__init__(*args, **kwargs)
        self.round = 0
        self.base_in_dim = base_in_dim
        self.base_num_classes = base_num_classes

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, agg_res = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is None:
            return aggregated_parameters, agg_res
        ndarrays = parameters_to_ndarrays(aggregated_parameters)
        base_model = get_model(in_dim=self.base_in_dim, num_classes=self.base_num_classes)
        keys = list(base_model.state_dict().keys())
        param_dict_np = {k: arr for k, arr in zip(keys, ndarrays)}
        try:
            set_state_dict_from_numpy_by_keys(base_model, param_dict_np)
            out = MODELS_DIR / f'global_round_{self.round}.pth'
            save_model_torch(base_model, str(out))
            try:
                mlflow.log_artifact(str(out), artifact_path=f'round_{self.round}')
                mlflow.log_metric('round_saved', self.round)
            except Exception:
                pass
            print(f'[server] saved global model {out}')
        except Exception as e:
            print('[server] save error:', e)
        self.round += 1
        return aggregated_parameters, agg_res



def run_flower_server(rounds):
    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI')
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(os.environ.get('MLFLOW_EXPERIMENT','FL_Experiment'))

    strategy = SaveModelFedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        base_in_dim=int(os.environ.get('IN_DIM',561)),
        base_num_classes=int(os.environ.get('NUM_CLASSES',6))
    )

    # Use FL_PORT env var or default 8080
    FL_PORT = int(os.environ.get('FL_PORT', 8080))
    print(f'[server] Flower server will run on port {FL_PORT}')
    
    config = ServerConfig(num_rounds=rounds)
    print(f'[server] starting Flower server on 0.0.0.0:{FL_PORT}')
    fl.server.start_server(server_address=f'0.0.0.0:{FL_PORT}', strategy=strategy, config=config)

if __name__ == '__main__':
    # Create initial global model if not present
    in_dim = int(os.environ.get('IN_DIM', 561))
    num_classes = int(os.environ.get('NUM_CLASSES', 6))
    initial_model_path = MODELS_DIR / 'global_round_0.pth'
    if not initial_model_path.exists():
        base_model = get_model(in_dim=in_dim, num_classes=num_classes)
        save_model_torch(base_model, str(initial_model_path))
        print('[server] created initial global model', initial_model_path)

    # Start control API (Flask) in background
    t = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5001), daemon=True)
    t.start()

    rounds = int(os.environ.get('FL_ROUNDS', 0))
    if rounds > 0:
        print(f"[server] FL_ROUNDS={rounds}, starting Flower server automatically")
        run_flower_server(rounds)
    else:
        print('[server] control API up on :5001 - call /start_rounds to run FL')
        t.join()
