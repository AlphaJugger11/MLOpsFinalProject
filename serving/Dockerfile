# FROM python:3.10-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . /app
# EXPOSE 8000
# CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]


FROM python:3.11
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY serving /app/serving
COPY shared /app/shared

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
