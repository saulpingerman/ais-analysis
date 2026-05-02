#!/bin/bash
# AIS Data Fetcher - shell wrapper for download_ais_data.py
#
# Usage:
#   ./fetch_ais_data.sh 2024-01-01 2024-01-31    # Download January 2024
#   ./fetch_ais_data.sh 2020-01-01 2020-12-31    # Download year 2020

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/download_ais_data.py"

show_usage() {
    cat <<EOF
Usage: $0 <start-date> <end-date> [options]

Arguments:
  start-date    Start date in YYYY-MM-DD format
  end-date      End date in YYYY-MM-DD format

Options:
  --force                Force redownload existing files
  --raw-dir PATH         Override raw_dir from config
  --config PATH          Configuration file path

Examples:
  $0 2024-01-01 2024-01-31
  $0 2020-01-01 2020-12-31
  $0 2024-01-01 2024-01-01 --force
  $0 2023-06-01 2023-06-30 --raw-dir /tmp/ais

Note: monthly files for pre-March 2024, daily files thereafter.
EOF
}

if [ $# -lt 2 ]; then
    echo "Error: missing required arguments"
    echo ""
    show_usage
    exit 1
fi

START_DATE="$1"
END_DATE="$2"
shift 2

if ! [[ "$START_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "Error: invalid start date format: $START_DATE (use YYYY-MM-DD)"
    exit 1
fi

if ! [[ "$END_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "Error: invalid end date format: $END_DATE (use YYYY-MM-DD)"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

echo "Starting AIS data download..."
echo "  Date range: $START_DATE to $END_DATE"
echo ""

python3 "$PYTHON_SCRIPT" --start-date "$START_DATE" --end-date "$END_DATE" "$@"
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "Download completed."
    echo "Run the pipeline with:  python3 run_pipeline.py"
else
    echo ""
    echo "Download failed with exit code: $exit_code"
fi

exit $exit_code
