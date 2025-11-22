import os

# ----------------------------
# Project Structure Definition
# ----------------------------
PROJECT_DIRS = [
    "airflow/dags",
    "src/client",
    "src/server",
    "src/models",
    "src/pipeline",
    "src/utils",
    "src/monitoring",
    "federated_clients",
]

PLACEHOLDER_FILES = {
    "airflow/dags/fl_training_dag.py": "# Airflow DAG Placeholder\n",
    "src/client/client.py": "# Client logic placeholder\n",
    "src/client/local_training.py": "# Local training placeholder\n",
    "src/server/server.py": "# Federated server placeholder\n",
    "src/server/aggregation.py": "# Aggregation logic placeholder\n",
    "src/server/model_dispatcher.py": "# Model dispatcher placeholder\n",
    "src/models/model_tiny.py": "# Tiny model placeholder\n",
    "src/models/model_medium.py": "# Medium model placeholder\n",
    "src/models/model_large.py": "# Large model placeholder\n",
    "src/pipeline/airflow_pipeline.py": "# Pipeline placeholder\n",
    "src/utils/config.py": "# Config placeholder\n",
    "src/utils/logger.py": "# Logger placeholder\n",
    "src/utils/dataset_loader.py": "# Dataset loader placeholder\n",
    "src/monitoring/drift_detection.py": "# Drift detection placeholder\n",
    "src/monitoring/mlflow_setup.py": "# MLflow setup placeholder\n",
    "src/monitoring/prometheus_exporter.py": "# Prometheus exporter placeholder\n",
}

# ----------------------------
# Create directories
# ----------------------------
def create_directories():
    for d in PROJECT_DIRS:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

# ----------------------------
# Create placeholder files
# ----------------------------
def create_placeholder_files():
    for filepath, content in PLACEHOLDER_FILES.items():
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Created file: {filepath}")

# ----------------------------
# Remove old federated client files
# ----------------------------
def clear_old_clients():
    folder = "federated_clients"
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isfile(path):
            os.remove(path)
            print(f"Removed old file: {path}")

# ----------------------------
# Generate 3 Federated Clients
# ----------------------------
def create_clients(num_clients=3):
    for i in range(1, num_clients + 1):
        file_path = f"federated_clients/client_{i}.txt"
        with open(file_path, "w") as f:
            f.write(f"Client {i} placeholder (no dataset stored here).")
        print(f"Created client placeholder: {file_path}")

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    print("Setting up project structure...\n")
    create_directories()
    create_placeholder_files()
    clear_old_clients()
    create_clients(3)
    print("\nSetup complete!")
