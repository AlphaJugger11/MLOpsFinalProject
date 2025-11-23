
# serving/Dockerfile
# FROM python:3.11-slim

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY shared/ /app/shared
# COPY models/ /app/models
# COPY serving/ /app/serving

# ENV PYTHONPATH=/app

# ENTRYPOINT ["python", "serving/app.py"]

# serving/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared utilities and models
COPY shared/ /app/shared
COPY models/ /app/models
COPY serving/ /app/serving

# Ensure imports work
ENV PYTHONPATH=/app

# Run FastAPI app
ENTRYPOINT ["python", "serving/app.py"]
