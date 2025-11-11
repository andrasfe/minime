FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PostgreSQL client libraries
RUN apt-get update && apt-get install -y \
    postgresql-client \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY mcp_server.py .
COPY ai_providers.py .
COPY .env.example .env.example
# Copy .env file (must exist in project root)
COPY .env .env

# Create directory for persistent data (if needed)
RUN mkdir -p /app/data

# Expose port (if using HTTP transport)
EXPOSE 8000

# Run the MCP server
CMD ["python", "mcp_server.py"]

