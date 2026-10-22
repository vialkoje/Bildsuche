FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/models \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# CPU-only PyTorch (spart ~2 GB CUDA-Ballast), danach der Rest
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install open_clip_torch transformers sentencepiece \
                pillow pillow-heif rawpy numpy pyyaml

WORKDIR /app
COPY app/bildsuche.py /app/bildsuche.py

CMD ["python", "/app/bildsuche.py"]
