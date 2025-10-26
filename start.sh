#!/bin/bash

# AI Browser Agent - Quick Start Script

set -e

echo "=================================="
echo "AI Browser Agent - Quick Start"
echo "=================================="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Installing..."
    brew install ollama
fi

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "🚀 Starting Ollama server..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 3
    echo "✓ Ollama server started (PID: $OLLAMA_PID)"
else
    echo "✓ Ollama is already running"
fi

# Check if llama3.2-vision model exists
echo ""
echo "📦 Checking for llama3.2-vision:11b model..."
if ! ollama list | grep -q "llama3.2-vision:11b"; then
    echo "⬇️  Downloading llama3.2-vision:11b (this may take a while, ~8GB)..."
    ollama pull llama3.2-vision:11b
    echo "✓ Model downloaded"
else
    echo "✓ llama3.2-vision:11b model already installed"
fi

# Check Python environment
echo ""
echo "🐍 Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
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

# Check ChromeDriver
echo ""
echo "🌐 Checking ChromeDriver..."
if ! command -v chromedriver &> /dev/null; then
    echo "⚠️  ChromeDriver not found. Installing..."
    brew install chromedriver
fi
echo "✓ ChromeDriver found: $(which chromedriver)"

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
python app.py
