# Base Image: Start with a slim Python image that already includes Python3.
FROM python:3.11-slim

# Set Working Directory.
WORKDIR /app

# Copy all files from local workspace into docker container
COPY . .
RUN ls .

RUN pip install --no-cache-dir -r requirements.txt
