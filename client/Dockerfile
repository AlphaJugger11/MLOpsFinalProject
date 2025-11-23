# FROM python:3.10-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . /app
# CMD ["python", "client/client.py", "--client_id", "client_1", "--data", "federated_clients/client_1.csv", "--fl_server", "server:8080"]


FROM python:3.11
WORKDIR /app

COPY client/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY client /app/client
COPY shared /app/shared

CMD ["python", "client/client.py", "--client_id", "${CLIENT_ID}", "--data", "${DATA_PATH}"]
