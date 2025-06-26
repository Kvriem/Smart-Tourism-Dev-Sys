#!/bin/bash

echo "=== Fixing Airflow Installation ==="

# Ensure we're in the airflow directory
cd /home/kariem/airflow

# Activate virtual environment
source airflow_venv/bin/activate

echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Remove all airflow packages to start fresh
echo "Removing existing Airflow packages..."
pip uninstall -y apache-airflow apache-airflow-providers-standard apache-airflow-core apache-airflow-task-sdk apache-airflow-providers-postgres apache-airflow-providers-smtp apache-airflow-providers-common-compat 2>/dev/null

# Clear pip cache
pip cache purge

# Install compatible Airflow version
echo "Installing Apache Airflow 2.8.1..."
export AIRFLOW_VERSION=2.8.1
export PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1,2)"
export CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

pip install --upgrade pip
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# Install compatible providers
echo "Installing compatible providers..."
pip install "apache-airflow-providers-postgres==5.8.0"
pip install "apache-airflow-providers-smtp==1.4.0"

# Install additional dependencies
echo "Installing additional dependencies..."
pip install selenium==4.15.0
pip install webdriver-manager>=3.8.0
pip install user-agent>=0.1.10
pip install pandas>=1.5.0
pip install requests>=2.28.0
pip install beautifulsoup4>=4.11.0
pip install psycopg2-binary>=2.9.0

# Set AIRFLOW_HOME
export AIRFLOW_HOME=/home/kariem/airflow

# Initialize the database
echo "Initializing Airflow database..."
airflow db init

# Create a user
echo "Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

echo "=== Installation completed! ==="
echo "You can now run: airflow standalone"
