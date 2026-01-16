FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    ffmpeg \
    libchromaprint-tools \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "main.py"]
