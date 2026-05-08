FROM python:3.11-slim

# Install GDAL system dependencies needed by geopandas / duckdb spatial
RUN apt-get update -q && \
    apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
        libspatialindex-dev \
        gcc \
        g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Force unbuffered Python output so logs appear immediately in HF Spaces
ENV PYTHONUNBUFFERED=1

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements-dashboard.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY scripts/ scripts/

# HF Spaces runs containers as a non-root user (uid 1000)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

CMD ["python", "app.py"]