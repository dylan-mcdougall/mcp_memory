FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the MCP server
COPY server.py /app/server.py

# Create memory storage directory
RUN mkdir -p /data/memories

# Environment variable for memory path
ENV MEMORY_BASE_PATH=/data/memories

# Expose the SSE port
EXPOSE 8080

# Run the MCP server with uvicorn
CMD ["python", "/app/server.py"]
