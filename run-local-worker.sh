#!/bin/bash

# Simple local Python worker script
set -e

echo "🐍 Setting up local Python worker..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  No virtual environment detected. Recommendations:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -e ."
    echo ""
    echo "Continuing anyway..."
fi

# Set environment variables
export AGNT5_COORDINATOR_ENDPOINT="http://localhost:9091"

echo "🔨 Building Rust extension..."
maturin develop --release

if [ $? -ne 0 ]; then
    echo "❌ Failed to build Rust extension"
    echo "💡 Make sure you have Rust installed: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

echo "✅ Rust extension built successfully"

echo "🎯 Starting Python worker:"
echo "  - Coordinator: http://localhost:9091"
echo "  - Service: my-python-service"
echo ""
echo "Press Ctrl+C to stop"

python examples/function_worker.py