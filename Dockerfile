FROM python:3.13-slim AS base

# OS依存パッケージ（Pillowなどで必要）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg-dev \
        zlib1g-dev \
        golang \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# ONNX Runtime (CPU版) をインストール
RUN curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz \
    -o /tmp/ort.tgz && \
    tar -xzf /tmp/ort.tgz -C /tmp && \
    cp /tmp/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so* /usr/local/lib/ && \
    mkdir -p /usr/local/include/onnxruntime && \
    cp -r /tmp/onnxruntime-linux-x64-1.23.2/include/* /usr/local/include/onnxruntime/ && \
    rm -rf /tmp/ort.tgz /tmp/onnxruntime-linux-x64-1.23.2 && \
    ldconfig

COPY . /workspace
WORKDIR /workspace

# Python依存
RUN pip install --no-cache-dir --requirement /workspace/app/py/requirements.txt

# ---- 開発用ステージ（Dev Container用） ----
FROM base AS dev
ENV env=develop
# dev だけに git を入れる
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --requirement /workspace/app/py/requirements_dev.txt
CMD ["bash"]

# ---- 本番用ステージ（Release Container用） ----
FROM base AS prod
ENV env=release
CMD ["bash"]
