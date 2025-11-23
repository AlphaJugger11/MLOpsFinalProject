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
