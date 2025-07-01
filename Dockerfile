# Use a lightweight Python base
FROM python:3.11-slim
# Set working directory
WORKDIR /app
# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    curl \
 && rm -rf /var/lib/apt/lists/*
# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# Copy your app code
COPY . .
# Expose Flask port
EXPOSE 5000
# Run the app
CMD ["python", "app.py"]