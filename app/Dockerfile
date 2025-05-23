FROM continuumio/miniconda3:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY app/environment.yml .

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "chestmodal", "/bin/bash", "-c"]

COPY app/ .
COPY fusion_model.py .
COPY config.py .
COPY mimic_fusion_model/ ./mimic_fusion_model/

RUN mkdir -p /app/models

EXPOSE 8000

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["conda", "run", "--no-capture-output", "-n", "chestmodal", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]