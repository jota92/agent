#!/bin/bash

# AI Browser Agent - Quick Start Script

set -e

echo "=================================="
echo "AI Browser Agent - Quick Start"
echo "=================================="
echo ""

# Python environment
echo ""
echo "🐍 Setting up Python environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    set +e
    python3 -m venv venv 2>/dev/null
    status=$?
    set -e
    if [ $status -ne 0 ]; then
        echo "⚠️  python3-venv が利用できません。virtualenv を利用します..."
        python3 -m pip install --user --upgrade virtualenv > /dev/null 2>&1
        python3 -m virtualenv venv
    fi
    echo "✓ Virtual environment created"
fi

source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"

# Preload local VLM weights (downloads on first run)
echo ""
echo "🧠 Preparing local vision-language model..."
python3 - <<'PY'
from app import load_vlm, vlm_device

try:
    load_vlm()
    print(f"✓ Vision-language model ready on device: {vlm_device}")
except Exception as exc:
    print(f"⚠️  Could not preload the vision-language model: {exc}")
    print("   The application will attempt to load it on first use.")
PY

# Check ChromeDriver
echo ""
echo "🌐 Checking ChromeDriver..."
if ! command -v chromedriver &> /dev/null; then
    echo "⚠️  ChromeDriver not found. Install ChromeDriver matching your Chrome/Chromium build before running autonomous tasks."
else
    echo "✓ ChromeDriver found: $(which chromedriver)"
fi

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "🚀 Starting AI Browser Agent..."
echo ""
echo "📍 Access at: http://127.0.0.1:5000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the app
python3 app.py
