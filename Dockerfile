# Base Image: Start with an image that already includes PyTorch and Cuda.
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# Set Working Directory.
WORKDIR /app

# Copy the code.
COPY ./ ./

RUN pip install --no-cache-dir .
# Install additional requirements for running demos.
RUN pip install --no-cache-dir -r ./demo/lightning/text-based/distributed/requirements.txt
