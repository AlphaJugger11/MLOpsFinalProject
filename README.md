# mlops-fl-project — Simplified single-model FedAvg skeleton

## Quick start (local)

1. Create a python venv and install deps:
   ```
   python -m venv venv
   source venv/bin/activate      # (or venv\Scripts\activate on Windows)
   pip install -r requirements.txt
   ```

2. Ensure you have the dataset split into federated clients:
   - `federated_clients/client_1.csv`
   - `federated_clients/client_2.csv`
   - `federated_clients/client_3.csv`
   (Do **not** commit these CSVs to git. Add federated_clients/ to .gitignore.)

3. Start server (control API and, optionally, auto-run rounds):
   ```
   python server/server.py
   # Or to auto-run rounds set FL_ROUNDS env var:
   FL_ROUNDS=3 IN_DIM=561 NUM_CLASSES=6 python server/server.py
   ```

4. Start 3 clients (each in separate terminal):
   ```
   python client/client.py --client_id client_1 --data federated_clients/client_1.csv
   python client/client.py --client_id client_2 --data federated_clients/client_2.csv
   python client/client.py --client_id client_3 --data federated_clients/client_3.csv
   ```

5. Trigger FL rounds (if server not auto-started):
   ```
   curl -X POST -H "Content-Type: application/json" -d '{"rounds":3}' http://127.0.0.1:5001/start_rounds
   ```

6. Or run with Docker Compose:
   ```
   docker-compose up --build
   ```

## Notes
- This simplified flow uses ONE single model architecture for all clients (FedAvg compatible).
- Server saves models to ./models/global_round_{r}.pth after each aggregation round.
- Monitoring, Airflow DAG and k8s manifests are skeletons for extension.
