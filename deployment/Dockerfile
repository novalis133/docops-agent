FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd -m -u 1000 docops && chown -R docops:docops /app
USER docops

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden)
CMD ["uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
