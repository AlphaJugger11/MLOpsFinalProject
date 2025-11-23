FROM apache/airflow:2.8.1
USER root
RUN pip install mlflow requests
USER airflow
COPY airflow/dags /opt/airflow/dags
