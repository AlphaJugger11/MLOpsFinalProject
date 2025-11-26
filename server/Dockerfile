# FROM python:3.10-slim
# WORKDIR /app
# COPY server/requirements.txt .
# RUN pip install --no-cache-dir -r server/requirements.txt
# COPY . /app
# EXPOSE 8080 5001
# CMD ["python", "server/server.py"]

FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY server/ /app/server
COPY shared/ /app/shared
COPY models/ /app/models


RUN touch /app/server/__init__.py



ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "server.server"]
