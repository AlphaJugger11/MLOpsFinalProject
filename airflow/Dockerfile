# FROM apache/airflow:2.8.1
# USER root
# RUN pip install mlflow requests
# USER airflow
# COPY airflow/dags /opt/airflow/dags


# FROM apache/airflow:2.8.1
# USER roo
# RUN pip install mlflow requests
# USER airflow

# # Corrected COPY path
# COPY dags /opt/airflow/dags

# Base Airflow image
FROM apache/airflow:2.8.1

# Set working directory
WORKDIR /opt/airflow

# Install Python packages
RUN pip install --no-cache-dir \
    mlflow==3.6.0 \
    requests==2.32.5

# Copy DAGs
COPY dags/ /opt/airflow/dags/

# Copy plugins if the folder exists
# This avoids errors if plugins/ does not exist
COPY plugins/ /opt/airflow/plugins/ 2>/dev/null || echo "No plugins folder found, skipping..."

# Expose webserver port
EXPOSE 8080

# Set default command
CMD ["webserver"]
