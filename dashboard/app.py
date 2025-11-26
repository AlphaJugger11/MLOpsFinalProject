import streamlit as st
import requests
import mlflow
from prometheus_api_client import PrometheusConnect
from kubernetes import client, config

CONTROL_API = "http://server:5001"
MLFLOW_URI = "http://server:5000"

st.set_page_config(layout="wide")
st.title("Federated Learning MLOps Dashboard")

# -----------------------------
# SECTION 1: SYSTEM STATUS
# -----------------------------
st.header("System Status")

if st.button("Check Server Status"):
    r = requests.get(f"{CONTROL_API}/get_fl_port")
    st.success(f"Flower Server Running on Port: {r.json()['fl_port']}")

# -----------------------------
# SECTION 2: START TRAINING
# -----------------------------
st.header("Trigger Federated Training")

rounds = st.number_input("Number of FL Rounds", min_value=1, max_value=20, value=3)

if st.button("Start Training"):
    r = requests.post(f"{CONTROL_API}/start_rounds", json={"rounds": rounds})
    st.success("FL Training Started")

# -----------------------------
# SECTION 3: CLIENT METRICS (PROMETHEUS)
# -----------------------------
st.header("Client Accuracy Metrics")

try:
    prom = PrometheusConnect(url="http://prometheus:9090", disable_ssl=True)
    metric = prom.get_current_metric_value(metric_name="client_local_accuracy")
    st.json(metric)
except:
    st.warning("Prometheus not reachable")

# -----------------------------
# SECTION 4: MODEL VERSIONS (MLflow)
# -----------------------------
st.header("Model Versions (MLflow)")

mlflow.set_tracking_uri(MLFLOW_URI)
experiments = mlflow.search_experiments()

for exp in experiments:
    st.subheader(exp.name)
    runs = mlflow.search_runs([exp.experiment_id])
    st.dataframe(runs[["run_id", "metrics.round_saved"]])

# -----------------------------
# SECTION 5: DEPLOY MODEL TO K8s
# -----------------------------
st.header("Deploy Latest Model to Kubernetes")

if st.button("Deploy Latest Model"):
    config.load_incluster_config()
    api = client.AppsV1Api()
    st.success("Kubernetes deployment triggered!")

# -----------------------------
# SECTION 6: INFERENCE TESTING
# -----------------------------
st.header("Inference Test")

payload = st.text_area("Input Feature Vector (JSON)", value='{"features":[0.0]*561}')

if st.button("Run Prediction"):
    r = requests.post("http://serving:8000/predict", json=eval(payload))
    st.success(r.json())
