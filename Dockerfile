# =============================================================================
# Dockerfile for Movie Rating Prediction API
# DDM501 - Lab 3: Testing & CI/CD
# =============================================================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for cache optimization)
COPY requirements.txt .

# Pre-install build dependencies for scikit-surprise
RUN pip install --upgrade pip "setuptools<70" wheel && \
    pip install numpy==1.26.2 Cython && \
    pip install scikit-surprise==1.1.3 --no-build-isolation

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
