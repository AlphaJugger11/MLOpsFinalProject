# server/server.py
import threading, os, mlflow
from pathlib import Path
from flask import Flask, request, jsonify
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays
from shared.model import get_model
from shared.model_utils import set_state_dict_from_numpy_by_keys, save_model_torch

app = Flask(__name__)
CLIENT_REGISTRY = {}
MODELS_DIR = Path.cwd() / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def assign_profile(payload):
    if payload.get('has_gpu'):
        return 'large'
    if payload.get('ram_gb',0) >= 8:
        return 'medium'
    return 'tiny'

@app.route('/register_profile', methods=['POST'])
def register_profile():
    payload = request.json or {}
    cid = payload.get('client_id')
    if not cid:
        return jsonify({'error':'client_id required'}), 400
    profile = assign_profile(payload)
    payload['assigned_profile'] = profile
    CLIENT_REGISTRY[cid] = payload
    return jsonify({'assigned_profile': profile})

@app.route('/list_clients', methods=['GET'])
def list_clients():
    return jsonify(CLIENT_REGISTRY)

@app.route('/start_rounds', methods=['POST'])
def start_rounds():
    data = request.json or {}
    rounds = int(data.get('rounds', 1))
    t = threading.Thread(target=lambda: run_flower_server(rounds), daemon=True)
    t.start()
    return jsonify({'status':'started','rounds': rounds})

class SaveModelFedAvg(FedAvg):
    def __init__(self, base_profile='medium', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_profile = base_profile
        self.round = 0

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, agg_res = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is None:
            return aggregated_parameters, agg_res
        ndarrays = parameters_to_ndarrays(aggregated_parameters)
        base_model = get_model(self.base_profile,
                               in_dim=int(os.environ.get('IN_DIM',561)),
                               num_classes=int(os.environ.get('NUM_CLASSES',6)))
        keys = list(base_model.state_dict().keys())
        param_dict_np = {}
        for k, arr in zip(keys, ndarrays):
            param_dict_np[k] = arr
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
    strategy = SaveModelFedAvg(base_profile=os.environ.get('BASE_PROFILE','medium'),
                               fraction_fit=1.0, fraction_eval=1.0,
                               min_fit_clients=3, min_eval_clients=3, min_available_clients=3)
    print('[server] starting Flower server on 0.0.0.0:8080')
    fl.server.start_server(server_address='0.0.0.0:8080', config={'num_rounds': rounds}, strategy=strategy)

if __name__ == '__main__':
    in_dim = int(os.environ.get('IN_DIM',561))
    num_classes = int(os.environ.get('NUM_CLASSES',6))
    default_profile = os.environ.get('BASE_PROFILE','medium')
    initial_model_path = MODELS_DIR / 'global_round_0.pth'
    if not initial_model_path.exists():
        base_model = get_model(default_profile, in_dim=in_dim, num_classes=num_classes)
        save_model_torch(base_model, str(initial_model_path))
        print('[server] created initial global model', initial_model_path)
    t = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5001), daemon=True)
    t.start()
    rounds = int(os.environ.get('FL_ROUNDS', 0))
    if rounds > 0:
        run_flower_server(rounds)
    else:
        print('[server] control API up on :5001 - call /start_rounds to run FL')
        t.join()
