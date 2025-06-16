#!/bin/bash
# Airflow Virtual Environment Activation Script
# Usage: source activate_venv.sh

echo "🚀 Activating Airflow virtual environment..."
source /home/kariem/airflow/airflow_venv/bin/activate

echo "✅ Virtual environment activated!"
echo "🔍 Python path: $(which python)"
echo "📦 Python version: $(python --version)"
echo "🌪️  Airflow version: $(python -c "import airflow; print(airflow.__version__)" 2>/dev/null || echo "Not available")"
echo ""
echo "📝 Available commands:"
echo "  airflow --help           # Show Airflow help"
echo "  airflow db init          # Initialize Airflow database"
echo "  airflow webserver        # Start Airflow webserver"
echo "  airflow scheduler        # Start Airflow scheduler"
echo ""
echo "💡 To deactivate the virtual environment, type: deactivate"
