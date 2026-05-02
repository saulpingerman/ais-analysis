#!/bin/bash
# AIS Pipeline Setup Script

set -e

echo "Setting up AIS Data Processing Pipeline..."

if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python3."
    exit 1
fi

echo "Python3 found"

if ! command -v pip3 &> /dev/null; then
    echo "Installing pip..."
    curl -s https://bootstrap.pypa.io/get-pip.py | python3
fi

echo "Pip available"

echo "Installing Python dependencies..."
python3 -m pip install -r requirements.txt
echo "Dependencies installed"

echo ""
echo "Setup complete."
echo ""
echo "Next steps:"
echo "1. Adjust paths in config/production.yaml if needed (defaults: ~/data/ais/{raw,clean,state})"
echo "2. Download a day to smoke-test:  python3 scripts/download_ais_data.py --start-date 2024-01-15 --end-date 2024-01-15"
echo "3. Run the pipeline on it:        python3 run_pipeline.py --max-files 1"
