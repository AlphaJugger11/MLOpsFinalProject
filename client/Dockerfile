FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
CMD ["python", "client/client.py", "--client_id", "client_1", "--data", "federated_clients/client_1.csv", "--fl_server", "server:8080"]
