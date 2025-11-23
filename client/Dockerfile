FROM python:3.11-slim

WORKDIR /app

# Set environment variables with defaults
ENV CLIENT_ID=1 \
    S3_BUCKET=mlopsprojbucket \
    USE_S3=true \
    AWS_REGION=us-east-1 \
    FL_SERVER=server:8080 \
    DATA_PATH=/app/federated_clients/client_1.csv

# Copy and install dependencies (including boto3)
COPY client/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY client /app/client
COPY shared /app/shared

# Set entry point with dynamic arguments
ENTRYPOINT ["sh", "-c", "python client/client.py --client_id ${CLIENT_ID} --data ${DATA_PATH} --fl_server ${FL_SERVER}"]
