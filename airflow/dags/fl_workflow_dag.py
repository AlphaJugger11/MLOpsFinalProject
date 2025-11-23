# # airflow/dags/fl_workflow_dag.py
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import datetime, timedelta
# import requests

# default_args = {
#     'owner': 'mlops',
#     'retries': 1,
#     'retry_delay': timedelta(minutes=2),
# }

# def trigger_fl_round(**context):
#     r = requests.post('http://server:5001/start_rounds', json={'rounds': 1})
#     return r.status_code

# def drift_check(**context):
#     # placeholder call - integrate Evidently here
#     return 'no-drift'

# with DAG(dag_id='fl_workflow', start_date=datetime(2025,1,1), schedule_interval=None,
#          default_args=default_args, catchup=False) as dag:
#     profile = PythonOperator(task_id='client_profiling', python_callable=lambda: 'noop - profiling removed')
#     start_round = PythonOperator(task_id='start_fl_round', python_callable=trigger_fl_round)
#     drift = PythonOperator(task_id='drift_detection', python_callable=drift_check)
#     profile >> start_round >> drift
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests

def trigger_federated_learning():
    url = "http://server:5001/start_rounds"
    res = requests.post(url, json={"rounds": 3})
    print("Triggered FL:", res.json())

def deploy_latest_model():
    print("Model deployed (K8S integration placeholder).")

with DAG(
    dag_id="federated_learning_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    start_fl = PythonOperator(
        task_id="start_fl_training",
        python_callable=trigger_federated_learning
    )

    deploy = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_latest_model
    )

    start_fl >> deploy
