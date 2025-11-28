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

# Install Playwright system dependencies (requires root)
RUN playwright install-deps chromium

# Create a non-root user with ID 1000 (standard for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Switch to the non-root user
USER user

# Install Playwright browsers (as user, so they end up in /home/user/.cache)
RUN playwright install chromium

# Copy the rest of the application code
COPY --chown=user:user . .

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
