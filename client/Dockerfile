FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system-level dependencies ONLY if needed
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY client/ /app/client
COPY shared/ /app/shared
COPY models/ /app/models
# Ensure 'client' is treated as a package
RUN touch /app/client/__init__.py

# Environment variables used at runtime
ENV CLIENT_ID=""
ENV DATA_PATH=""
ENV PYTHONPATH=/app

# ENTRYPOINT ensures args & env vars are passed correctly
ENTRYPOINT ["python", "-m", "client.client"]

# CMD supplies default arguments that can be overridden
CMD ["--client_id", "${CLIENT_ID}", "--data", "${DATA_PATH}"]
