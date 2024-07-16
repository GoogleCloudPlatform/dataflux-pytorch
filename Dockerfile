# Base Image: Start with a slim Python image that already includes Python3.
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# Set Working Directory.
WORKDIR /app

# Copy the Python training code and the requirements.txt file.
COPY ./ ./

RUN pip install .
