FROM python:3.11-slim

WORKDIR /app

# Set environment variables with defaults
ENV FL_ROUNDS=3 \
    IN_DIM=561 \
    NUM_CLASSES=6 \
    FL_PORT=8080 \
    MLFLOW_TRACKING_URI="" \
    MLFLOW_EXPERIMENT="FL_Experiment"

# Copy and install dependencies
COPY server/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY server /app/server
COPY shared /app/shared

# Create models directory
RUN mkdir -p /app/models

# Expose ports
EXPOSE 8080 5001

# Set entry point
CMD ["python", "server/server.py"]
