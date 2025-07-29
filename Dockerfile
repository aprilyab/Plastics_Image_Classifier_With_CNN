# Use official Python base image (not slim, more reliable for TensorFlow)
FROM python:3.10

# Set working directory
WORKDIR /app

# Install system dependencies required by TensorFlow and image processing libs
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

# Upgrade pip and install Python packages with more reliable flags
RUN pip install --upgrade pip \
 && pip install --timeout=100 --retries=10 --no-cache-dir -r requirements.txt

# Copy app files (including models/, app.py, etc.)
COPY . .

# Expose Gradio port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]

