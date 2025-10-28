# Use the base image
FROM agrigorev/zoomcamp-model:2025

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY pyproject.toml ./
COPY main.py .
COPY pipeline_v1.bin .

# Install dependencies from pyproject.toml
# Install dependencies from pyproject.toml
RUN pip install --no-cache-dir "scikit-learn==1.6.1" fastapi uvicorn

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
