FROM apache/airflow:latest

# Install selenium and any other dependencies
USER root
COPY requirements.txt .
RUN pip install --no-cache-dir selenium -r requirements.txt
COPY requirements.txt .
