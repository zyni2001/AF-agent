# Dockerfile for FOLIO Benchmark Green Agent (AgentBeats deployment)
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY main.py ./
COPY run.sh ./
COPY .env ./

# Install Python dependencies
RUN pip install --no-cache-dir \
    a2a-sdk[http-server]>=0.3.8 \
    litellm>=1.0.0 \
    pandas>=2.0.0 \
    typer>=0.9.0 \
    uvicorn>=0.27.0 \
    python-dotenv>=1.0.0

# Make run.sh executable
RUN chmod +x run.sh

# Expose port for the agent
EXPOSE 8080

# Set environment variables for AgentBeats
ENV HOST=0.0.0.0
ENV AGENT_PORT=8080

# Start the green agent
CMD ["./run.sh"]

