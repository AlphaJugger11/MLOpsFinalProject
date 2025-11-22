#!/usr/bin/env python3
"""
generate_full_pipeline.py
Creates a complete mlops-fl-project skeleton with:
 - server/
 - client/
 - shared/
 - serving/
 - airflow/dags
 - monitoring/
 - k8s/
 - docker-compose.yml
 - requirements.txt
 - README_RUN.md

Run from project root:
    python generate_full_pipeline.py
"""

import os
from pathlib import Path
from textwrap import dedent

ROOT = Path.cwd()

FILES = {
    # Shared model + utils
    "shared/model.py": dedent("""\
        # shared/model.py
        import torch.nn as nn

        class BaseNet(nn.Module):
            def __init__(self, in_dim=561, hidden_layers=[256,128], num_classes=6):
                super().__init__()
                layers = []
                prev = in_dim
                for h in hidden_layers:
                    layers.append(nn.Linear(prev, h))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.2))
                    prev = h
                layers.append(nn.Linear(prev, num_classes))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        def get_model(size='medium', in_dim=561, num_classes=6):
            size = (size or 'medium').lower()
            if size == 'tiny':
                return BaseNet(in_dim=in_dim, hidden_layers=[64], num_classes=num_classes)
            if size == 'large':
                return BaseNet(in_dim=in_dim, hidden_layers=[512,256,128], num_classes=num_classes)
            return BaseNet(in_dim=in_dim, hidden_layers=[256,128], num_classes=num_classes)
    """),

    "shared/model_utils.py": dedent("""\
        # shared/model_utils.py
        import torch, os
        from collections import OrderedDict

        def save_model_torch(model, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model.state_dict(), path)

        def load_model_torch(model, path, map_location=None):
            state = torch.load(path, map_location=map_location)
            model.load_state_dict(state)
            return model

        def get_parameters_numpy(model):
            return [val.cpu().numpy() for _, val in model.state_dict().items()]

        def set_parameters_from_numpy(model, parameters):
            state_dict = model.state_dict()
            new_state = OrderedDict()
            for (k, _), arr in zip(state_dict.items(), parameters):
                new_state[k] = torch.tensor(arr)
            model.load_state_dict(new_state)

        def set_state_dict_from_numpy_by_keys(model, param_dict_np):
            state = model.state_dict()
            for k in list(state.keys()):
                if k not in param_dict_np:
                    raise KeyError(f"Missing key: {k}")
                state[k] = torch.tensor(param_dict_np[k])
            model.load_state_dict(state)
    """),

    # Server
    "server/server.py": dedent("""\
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
    """),

    "server/requirements.txt": dedent("""\
        flwr==1.6.0
        flask
        mlflow
        torch
        numpy
    """),

    "server/Dockerfile": dedent("""\
        FROM python:3.10-slim
        WORKDIR /app
        COPY server/requirements.txt .
        RUN pip install --no-cache-dir -r server/requirements.txt
        COPY . /app
        EXPOSE 8080 5001
        CMD ["python", "server/server.py"]
    """),

    # Client
    "client/client.py": dedent("""\
        # client/client.py
        import argparse, requests, socket, psutil, os
        import torch, flwr as fl
        from pathlib import Path
        from shared.model_utils import get_parameters_numpy, set_parameters_from_numpy, load_model_torch
        from shared.model import get_model
        from shared.data_utils import load_client_csv

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def profile_and_register(server_profile_url, client_id):
            info = {
                'client_id': client_id,
                'cpu_cores': psutil.cpu_count(logical=False),
                'ram_gb': int(psutil.virtual_memory().total / (1024**3)),
                'has_gpu': torch.cuda.is_available(),
                'hostname': socket.gethostname()
            }
            r = requests.post(server_profile_url.rstrip('/') + '/register_profile', json=info, timeout=10)
            r.raise_for_status()
            return r.json()

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
                return get_parameters_numpy(self.model)

            def fit(self, parameters, config):
                set_parameters(self.model, parameters)
                self.model.train()
                epochs = int(config.get('local_epochs', 1))
                for _ in range(epochs):
                    for Xb, yb in self.train_loader:
                        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                        self.optimizer.zero_grad()
                        logits = self.model(Xb)
                        loss = self.criterion(logits, yb)
                        loss.backward()
                        self.optimizer.step()
                return get_parameters_numpy(self.model), len(self.train_loader.dataset), {}

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
                return float(loss/total) if total>0 else 0.0, total, {'accuracy': acc}

        def start_client(fl_server, control_api, client_id, data_path, local_epochs=1):
            try:
                reg = profile_and_register(control_api, client_id)
                profile = reg.get('assigned_profile','medium')
            except Exception as e:
                print('[client] register failed:', e)
                profile = 'medium'
            print(f'[{client_id}] profile: {profile}')
            train_loader, val_loader, in_dim, num_classes, _ = load_client_csv(data_path)
            model = get_model(profile, in_dim=in_dim, num_classes=num_classes)
            # try load latest global model
            models_dir = Path.cwd().parent / 'models'
            if not models_dir.exists():
                models_dir = Path.cwd().resolve().parents[2] / 'models'
            model_files = sorted(models_dir.glob('global_round_*.pth'))
            if model_files:
                latest = model_files[-1]
                try:
                    load_model_torch(model, str(latest), map_location=DEVICE)
                    print(f'[{client_id}] loaded global model {latest}')
                except Exception as e:
                    print('[client] load model error:', e)
            client = FlowerClient(model, train_loader, val_loader, client_id)
            fl.client.start_numpy_client(server_address=fl_server, client=client)

        if __name__ == '__main__':
            parser = argparse.ArgumentParser()
            parser.add_argument('--client_id', required=True)
            parser.add_argument('--data', required=True)
            parser.add_argument('--fl_server', default='127.0.0.1:8080')
            parser.add_argument('--control_api', default='http://127.0.0.1:5001')
            parser.add_argument('--local_epochs', type=int, default=1)
            args = parser.parse_args()
            start_client(args.fl_server, args.control_api, args.client_id, args.data, local_epochs=args.local_epochs)
    """),

    "client/data_prep.py": dedent("""\
        # client/data_prep.py
        import pandas as pd
        import numpy as np
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.preprocessing import LabelEncoder

        def load_client_csv(path, batch_size=64, val_split=0.2):
            df = pd.read_csv(path)
            if 'subject' in df.columns and 'activity' in df.columns:
                feature_cols = [c for c in df.columns if c not in ('subject','activity')]
                y_col = 'activity'
            else:
                feature_cols = df.columns[:-1].tolist()
                y_col = df.columns[-1]
            X = df[feature_cols].values.astype('float32')
            y = df[y_col].values
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            n = len(X)
            idx = int(n*(1-val_split))
            if idx <= 0:
                idx = max(1, int(n*0.8))
            X_train, X_val = X[:idx], X[idx:]
            y_train, y_val = y_enc[:idx], y_enc[idx:]
            train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long()),
                                      batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).long()),
                                    batch_size=256)
            in_dim = X.shape[1]
            num_classes = len(np.unique(y_enc))
            return train_loader, val_loader, in_dim, num_classes, le
    """),

    "client/Dockerfile": dedent("""\
        FROM python:3.10-slim
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt
        COPY . /app
        CMD ["python", "client/client.py", "--client_id", "client_1", "--data", "federated_clients/client_1.csv", "--fl_server", "server:8080", "--control_api", "http://server:5001"]
    """),

    # Serving
    "serving/app.py": dedent("""\
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
    """),

    "serving/Dockerfile": dedent("""\
        FROM python:3.10-slim
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt
        COPY . /app
        EXPOSE 8000
        CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
    """),

    # Monitoring
    "monitoring/prometheus_metrics.py": dedent("""\
        # monitoring/prometheus_metrics.py
        from prometheus_client import start_http_server, Gauge
        import time

        client_acc = Gauge('client_local_accuracy', 'Per-client validation accuracy', ['client_id'])
        global_round = Gauge('fl_global_round', 'Global FL round number')

        def start_metrics_server(port=9100):
            start_http_server(port)
            print(f'[metrics] Prometheus metrics exposed on :{port}')

        if __name__ == '__main__':
            start_metrics_server()
            r = 0
            while True:
                r += 1
                global_round.set(r)
                client_acc.labels(client_id='client_1').set(0.7)
                time.sleep(10)
    """),

    "monitoring/evidently_check.py": dedent("""\
        # monitoring/evidently_check.py
        import pandas as pd
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently import ColumnMapping

        def run_drift_check(reference_csv, current_csv, out_html='/tmp/drift_report.html'):
            ref = pd.read_csv(reference_csv)
            curr = pd.read_csv(current_csv)
            report = Report(metrics=[DataDriftPreset()])
            col_map = ColumnMapping()
            report.run(reference_data=ref, current_data=curr, column_mapping=col_map)
            report.save_html(out_html)
            return out_html

        if __name__ == '__main__':
            print(run_drift_check('reference.csv', 'current.csv'))
    """),

    # Airflow DAG
    "airflow/dags/fl_workflow_dag.py": dedent("""\
        # airflow/dags/fl_workflow_dag.py
        from airflow import DAG
        from airflow.operators.python import PythonOperator
        from datetime import datetime, timedelta
        import requests

        default_args = {
            'owner':'mlops',
            'retries':1,
            'retry_delay':timedelta(minutes=2)
        }

        def trigger_fl_round(**ctx):
            resp = requests.post('http://server:5001/start_rounds', json={'rounds':1})
            return resp.status_code

        def drift_check(**ctx):
            return 'ok'

        with DAG(dag_id='fl_workflow', start_date=datetime(2025,1,1),
                schedule_interval=None, default_args=default_args, catchup=False) as dag:

            profile = PythonOperator(task_id='client_profiling', python_callable=lambda: 'ok')
            start_round = PythonOperator(task_id='start_fl_round', python_callable=trigger_fl_round)
            drift = PythonOperator(task_id='drift_check', python_callable=drift_check)

            profile >> start_round >> drift
    """),

    # k8s manifests
    "k8s/server-deployment.yaml": dedent("""\
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: fl-server
        spec:
          replicas: 1
          selector:
            matchLabels:
              app: fl-server
          template:
            metadata:
              labels:
                app: fl-server
            spec:
              containers:
              - name: server
                image: yourrepo/fl-server:latest
                ports:
                - containerPort: 8080
                - containerPort: 5001
    """),

    "k8s/server-service.yaml": dedent("""\
        apiVersion: v1
        kind: Service
        metadata:
          name: fl-server-svc
        spec:
          selector:
            app: fl-server
          ports:
            - port: 8080
              targetPort: 8080
            - port: 5001
              targetPort: 5001
          type: LoadBalancer
    """),

    "k8s/client-deployment.yaml": dedent("""\
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: fl-client
        spec:
          replicas: 3
          selector:
            matchLabels:
              app: fl-client
          template:
            metadata:
              labels:
                app: fl-client
            spec:
              containers:
              - name: client
                image: yourrepo/fl-client:latest
                env:
                - name: FL_SERVER
                  value: "fl-server-svc:8080"
                - name: CONTROL_API
                  value: "http://fl-server-svc:5001"
                volumeMounts:
                - name: client-data
                  mountPath: /app/federated_clients
              volumes:
              - name: client-data
                hostPath:
                  path: /data/federated_clients
    """),

    # docker-compose + root requirements + README
    "docker-compose.yml": dedent("""\
        version: "3.8"
        services:
          server:
            build:
              context: .
              dockerfile: server/Dockerfile
            container_name: fl_server
            ports:
              - "8080:8080"
              - "5001:5001"
            volumes:
              - ./models:/app/models
              - ./federated_clients:/app/federated_clients
            environment:
              - FL_ROUNDS=3
              - BASE_PROFILE=medium
              - IN_DIM=561
              - NUM_CLASSES=6

          client1:
            build:
              context: .
              dockerfile: client/Dockerfile
            depends_on:
              - server
            volumes:
              - ./federated_clients:/app/federated_clients
              - ./models:/app/models
            command: ["python","client/client.py","--client_id","client_1","--data","federated_clients/client_1.csv","--fl_server","fl_server:8080","--control_api","http://fl_server:5001"]

          client2:
            build:
              context: .
              dockerfile: client/Dockerfile
            depends_on:
              - server
            volumes:
              - ./federated_clients:/app/federated_clients
              - ./models:/app/models
            command: ["python","client/client.py","--client_id","client_2","--data","federated_clients/client_2.csv","--fl_server","fl_server:8080","--control_api","http://fl_server:5001"]

          client3:
            build:
              context: .
              dockerfile: client/Dockerfile
            depends_on:
              - server
            volumes:
              - ./federated_clients:/app/federated_clients
              - ./models:/app/models
            command: ["python","client/client.py","--client_id","client_3","--data","federated_clients/client_3.csv","--fl_server","fl_server:8080","--control_api","http://fl_server:5001"]
    """),

    "requirements.txt": dedent("""\
        flwr==1.6.0
        torch>=1.13
        pandas
        scikit-learn
        numpy
        flask
        requests
        mlflow
        psutil
        prometheus-client
        evidently
        uvicorn
        fastapi
    """),

    "README_RUN.md": dedent("""\
        # Quick run instructions (local)

        ## 1) Generate files
        This repo already contains a generator script `generate_full_pipeline.py` to create the full skeleton.
        If you used that, files should exist.

        ## 2) Create federated client CSVs
        Ensure you have: `federated_clients/client_1.csv`, `client_2.csv`, `client_3.csv` (your existing data-splitting script produces them).

        ## 3) Python local run (no Docker)
        - Start server (control API + optionally Flower if FL_ROUNDS set):
          python server/server.py
          or to auto-run rounds:
          FL_ROUNDS=3 BASE_PROFILE=medium IN_DIM=561 NUM_CLASSES=6 python server/server.py

        - Start clients (three terminals):
          python client/client.py --client_id client_1 --data federated_clients/client_1.csv --fl_server 127.0.0.1:8080 --control_api http://127.0.0.1:5001
          python client/client.py --client_id client_2 --data federated_clients/client_2.csv --fl_server 127.0.0.1:8080 --control_api http://127.0.0.1:5001
          python client/client.py --client_id client_3 --data federated_clients/client_3.csv --fl_server 127.0.0.1:8080 --control_api http://127.0.0.1:5001

        - Trigger FL rounds:
          curl -X POST -H "Content-Type: application/json" -d '{"rounds":3}' http://127.0.0.1:5001/start_rounds

        ## 4) Docker Compose (recommended for integration testing)
        docker-compose up --build

        ## 5) Kubernetes
        Build & push images and apply manifests in k8s/.

        ## Notes
        - FL aggregation requires identical parameter shapes across clients for FedAvg.
        - For heterogeneous model architectures see 'distillation' or 'ensemble' approaches (advanced).
    """),
}

# extra small shared data_utils used by client code (to avoid duplication)
FILES["shared/data_utils.py"] = dedent("""\
    # shared/data_utils.py
    import pandas as pd
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import LabelEncoder

    def load_client_csv(path, batch_size=64, val_split=0.2):
        df = pd.read_csv(path)
        if 'subject' in df.columns and 'activity' in df.columns:
            feature_cols = [c for c in df.columns if c not in ('subject','activity')]
            y_col = 'activity'
        else:
            feature_cols = df.columns[:-1].tolist()
            y_col = df.columns[-1]
        X = df[feature_cols].values.astype('float32')
        y = df[y_col].values
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        n = len(X)
        idx = int(n*(1-val_split))
        if idx <= 0:
            idx = max(1, int(n*0.8))
        X_train, X_val = X[:idx], X[idx:]
        y_train, y_val = y_enc[:idx], y_enc[idx:]
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long()),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).long()),
                                batch_size=256)
        in_dim = X.shape[1]
        num_classes = len(np.unique(y_enc))
        return train_loader, val_loader, in_dim, num_classes, le
""")

def ensure_path(p: Path):
    d = p.parent
    d.mkdir(parents=True, exist_ok=True)

def write_files():
    for rel, content in FILES.items():
        path = ROOT / rel
        ensure_path(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created {rel}")

if __name__ == "__main__":
    print("Generating project files...")
    write_files()
    print("\\nDone. Files created. Next steps in README_RUN.md")
