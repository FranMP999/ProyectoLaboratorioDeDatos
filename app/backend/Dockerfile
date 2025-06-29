# Utiliza una imagen base con Python instalado
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y curl libgomp1
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U email-validator

COPY . .

RUN mkdir -p /data

ENV DATA_DIR=/data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
