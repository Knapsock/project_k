FROM python:3.11-slim

# Install system libraries for Tkinter, Pillow, etc.
RUN apt-get update && apt-get install -y \
    python3-tk \
    tcl \
    tk \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run your app (replace with your main Python file if not app.py)
CMD ["python", "app.py"]
