#  Copyright 2024 Google LLC

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#       https://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Base Image: Start with a slim Python image that already includes Python3.
FROM python:3.11-slim

# Set Working Directory.
WORKDIR /app

# Copy all files from local workspace into docker container
COPY . .
RUN ls .

RUN pip install --no-cache-dir -r requirements.txt
