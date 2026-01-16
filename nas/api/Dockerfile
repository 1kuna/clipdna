FROM python:3.11-slim

WORKDIR /app

# FFmpeg for signature verification
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV API_PORT=49321

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${API_PORT}"]
