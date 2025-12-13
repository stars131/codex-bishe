# Network Attack Detection - Deep Learning Environment
# 基于多源数据融合的网络攻击检测系统
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Network Attack Detection Project"
LABEL description="Multi-source data fusion based network attack detection system"
LABEL version="1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    wget \
    curl \
    libpcap-dev \
    tshark \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install streamlit plotly

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/logs \
    outputs/checkpoints outputs/results outputs/figures outputs/reports

# Make main.py executable
RUN chmod +x main.py

# Expose ports
# 8501: Streamlit Dashboard
# 8888: Jupyter Notebook
# 6006: TensorBoard
EXPOSE 8501 8888 6006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command - launch Streamlit dashboard
CMD ["streamlit", "run", "src/visualization/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
