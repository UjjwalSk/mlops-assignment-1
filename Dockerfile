FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install curl for docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for security compliance
RUN useradd -u 10001 -m -s /bin/bash appuser

# Copy source code and pre-trained artifacts
COPY src ./src
COPY artifacts ./artifacts

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.heart_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
