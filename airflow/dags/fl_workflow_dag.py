# airflow/dags/fl_workflow_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

def trigger_fl_round(**context):
    r = requests.post('http://server:5001/start_rounds', json={'rounds': 1})
    return r.status_code

def drift_check(**context):
    # placeholder call - integrate Evidently here
    return 'no-drift'

with DAG(dag_id='fl_workflow', start_date=datetime(2025,1,1), schedule_interval=None,
         default_args=default_args, catchup=False) as dag:
    profile = PythonOperator(task_id='client_profiling', python_callable=lambda: 'noop - profiling removed')
    start_round = PythonOperator(task_id='start_fl_round', python_callable=trigger_fl_round)
    drift = PythonOperator(task_id='drift_detection', python_callable=drift_check)
    profile >> start_round >> drift
