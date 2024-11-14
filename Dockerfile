# Base Image: Start with an image that already includes PyTorch and Cuda.
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# Set Working Directory.
WORKDIR /app

# Install additional requirements for running demos.
# Do this before copying the code so that these commands are still cached
# by Docker even if the code changes.
COPY ./benchmark/standalone_dataloader/requirements.txt requirements-1.txt
RUN pip install --no-cache-dir -r requirements-1.txt
COPY ./demo/lightning/text_based/distributed/requirements.txt requirements-2.txt
RUN pip install --no-cache-dir -r requirements-2.txt
COPY ./demo/lightning/image_segmentation/requirements.txt requirements-3.txt
RUN pip install --no-cache-dir -r requirements-3.txt
COPY ./demo/lightning/checkpoint/requirements.txt requirements-4.txt
RUN pip install --no-cache-dir -r requirements-4.txt

# Copy the code.
COPY ./ ./

RUN pip install --no-cache-dir .
