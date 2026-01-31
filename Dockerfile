# Multi-stage optimized build for fast cold starts
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in builder stage
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Final stage - minimal image
FROM python:3.10-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Set working directory
WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY models/.gitkeep ./models/.gitkeep

# Create directories with proper permissions
RUN mkdir -p models data/raw/ai data/raw/human

# Create non-root user
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app
USER apiuser

# Expose port
EXPOSE 8000

# Optimized health check (more frequent, faster timeout)
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=2 \
    CMD curl -f http://localhost:8000/health || exit 1

# Fast startup with pre-loading
CMD
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 75
