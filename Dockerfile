# Optimized Docker build for AI Voice Detection API
FROM python:3.10-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies globally (before switching to non-root user)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY models/.gitkeep ./models/.gitkeep

# Create directories with proper permissions
RUN mkdir -p models data/raw/ai data/raw/human

# Create non-root user and set ownership
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app

# Switch to non-root user
USER apiuser

# Expose port
EXPOSE 8000

# Optimized health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=2 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run deployment training and start application
CMD python scripts/deploy_train.py && exec uvicorn app.main:app --host 0.0.0.0 --port 8000
