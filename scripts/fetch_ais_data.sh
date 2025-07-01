#!/bin/bash
# 
# AIS Data Fetcher - Shell wrapper for download_ais_data.py
# 
# Usage:
#   ./fetch_ais_data.sh 2024-01-01 2024-01-31    # Download January 2024
#   ./fetch_ais_data.sh 2020-01-01 2020-12-31    # Download year 2020
#

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/download_ais_data.py"

# Function to show usage
show_usage() {
    echo "Usage: $0 <start-date> <end-date> [options]"
    echo ""
    echo "Arguments:"
    echo "  start-date    Start date in YYYY-MM-DD format"
    echo "  end-date      End date in YYYY-MM-DD format"
    echo ""
    echo "Options:"
    echo "  --force       Force redownload existing files"
    echo "  --bucket      S3 bucket name (overrides config)"
    echo ""
    echo "Examples:"
    echo "  $0 2024-01-01 2024-01-31                    # Download January 2024"
    echo "  $0 2020-01-01 2020-12-31                    # Download year 2020"
    echo "  $0 2024-01-01 2024-01-01 --force           # Force redownload single day"
    echo "  $0 2023-06-01 2023-06-30 --bucket my-bucket # Use specific bucket"
    echo ""
    echo "Note: Script automatically handles monthly (pre-2024) vs daily (2024+) files"
}

# Check arguments
if [ $# -lt 2 ]; then
    echo "❌ Error: Missing required arguments"
    echo ""
    show_usage
    exit 1
fi

START_DATE="$1"
END_DATE="$2"
shift 2

# Validate date format (basic check)
if ! [[ "$START_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "❌ Error: Invalid start date format: $START_DATE"
    echo "   Use YYYY-MM-DD format"
    exit 1
fi

if ! [[ "$END_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "❌ Error: Invalid end date format: $END_DATE"
    echo "   Use YYYY-MM-DD format"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found. Please install Python 3."
    exit 1
fi

echo "🚀 Starting AIS data download..."
echo "   Date range: $START_DATE to $END_DATE"
echo "   Script: $PYTHON_SCRIPT"
echo ""

# Run the Python script with all remaining arguments
python3 "$PYTHON_SCRIPT" --start-date "$START_DATE" --end-date "$END_DATE" "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ Download completed successfully!"
    echo "   Files are now available in your S3 bucket"
    echo "   You can now process them with: python3 scripts/s3_ais_processor.py"
else
    echo ""
    echo "❌ Download failed with exit code: $exit_code"
    echo "   Check the error messages above for details"
fi

exit $exit_code