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

# Use the official Apache Airflow image
FROM apache/airflow:2.8.1

# Switch to root to install OS packages if needed
USER root

# Install OS-level dependencies if required (optional)
# RUN apt-get update && apt-get install -y \
#     git \
#     curl \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# Switch back to the airflow user
USER airflow

# Install Python packages needed for your DAGs
RUN pip install --no-cache-dir \
    mlflow==3.6.0 \
    requests==2.32.5

# Set workdir for airflow dags/plugins (optional)
WORKDIR /opt/airflow

# Copy your DAGs into the image
# Adjust path as per your project structure
COPY dags/ /opt/airflow/dags/
COPY plugins/ /opt/airflow/plugins/

# Expose webserver port
EXPOSE 8080

# Entrypoint and CMD are already handled by airflow image
