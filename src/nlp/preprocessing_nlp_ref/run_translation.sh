#!/bin/bash

echo "🚀 Setting up GPU-Accelerated Translation Pipeline"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo "🎯 Starting translation pipeline..."
python translation_local_gpu.py

echo "✅ Translation pipeline completed!"
