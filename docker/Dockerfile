FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY docker/serve.py ./serve.py

EXPOSE 8080

ENTRYPOINT ["python", "serve.py"]