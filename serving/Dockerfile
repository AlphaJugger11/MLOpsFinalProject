# FROM python:3.10-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . /app
# EXPOSE 8000
# CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

# serving/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /app/shared
COPY models/ /app/models
COPY serving/ /app/serving

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "serving/app.py"]

