FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (for docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Create results directory
RUN mkdir -p results

# The container acts as an offline CLI runner
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
