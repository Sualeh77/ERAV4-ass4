# Dockerfile for Gradio MNIST Classifier

# Lambda-compatible Dockerfile
FROM public.ecr.aws/docker/library/python:3.12-slim

# Install AWS Lambda Web Adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the required Python modules
COPY config.py ./
COPY model.py ./
COPY gradio_app.py ./

# Copy the trained model
COPY models/ ./models/

# Expose the port Gradio will run on
EXPOSE 7860

# Set environment variables
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Command to run the application - Use python directly, not uv
CMD ["python", "gradio_app.py"]