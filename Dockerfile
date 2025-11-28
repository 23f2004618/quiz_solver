# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for common libraries
RUN apt-get update && apt-get install -y \
    git \
    wget \
    gnupg \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers and required system dependencies
# We install only chromium to save space, as the code uses p.chromium.launch()
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy the rest of the application code
COPY . .

# Create a non-root user with ID 1000 (standard for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Change ownership of the app directory to the user
RUN chown -R user:user /app

# Switch to the non-root user
USER user

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
