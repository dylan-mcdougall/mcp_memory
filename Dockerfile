FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install fastmcp pydantic

# Copy the MCP server
COPY server.py /app/server.py

# Create memory storage directory
RUN mkdir -p /data/memories

# Environment variable for memory path
ENV MEMORY_BASE_PATH=/data/memories

# Run the server
CMD ["tail", "-f", "/dev/null"]
