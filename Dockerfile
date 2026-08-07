# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default command to run FastAPI backend (or you can override it)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]