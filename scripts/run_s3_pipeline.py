#!/usr/bin/env python3
"""
Example script demonstrating how to use the S3 AIS processor.

This script shows different ways to run the AIS processing pipeline:
1. Basic processing with default settings
2. Processing with interpolation enabled
3. Processing a limited number of files for testing
4. Custom configuration settings
"""

import subprocess
import sys
from pathlib import Path

# Configuration
BUCKET_NAME = "ais-pipeline-data-10179bbf-us-east-1"
INPUT_PREFIX = "data/01_raw/ais_dk/raw/"
OUTPUT_PREFIX = "data/03_primary/cleaned_ais/"

def run_command(cmd):
    """Run a command and print the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
    else:
        print("❌ Error!")
        if result.stderr:
            print("Error:", result.stderr)
    
    return result.returncode == 0

def example_basic_processing():
    """Example 1: Basic processing with default settings."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Processing")
    print("="*60)
    
    cmd = [
        "python3", "s3_ais_processor.py",
        "--bucket", BUCKET_NAME,
        "--input-prefix", INPUT_PREFIX,
        "--output-prefix", OUTPUT_PREFIX,
        "--max-files", "1",  # Process just 1 file for testing
    ]
    
    return run_command(cmd)

def example_with_interpolation():
    """Example 2: Processing with interpolation enabled."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Processing with Interpolation")
    print("="*60)
    
    cmd = [
        "python3", "s3_ais_processor.py",
        "--bucket", BUCKET_NAME,
        "--input-prefix", INPUT_PREFIX,
        "--output-prefix", "data/03_primary/interpolated_ais/",
        "--max-files", "1",
        "--interpolate",
        "--interpolate-interval", "10",  # 10-minute intervals
    ]
    
    return run_command(cmd)

def example_custom_settings():
    """Example 3: Processing with custom speed and gap settings."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Speed and Gap Settings")
    print("="*60)
    
    cmd = [
        "python3", "s3_ais_processor.py",
        "--bucket", BUCKET_NAME,
        "--input-prefix", INPUT_PREFIX,
        "--output-prefix", "data/03_primary/custom_cleaned_ais/",
        "--max-files", "1",
        "--speed-thresh", "60.0",  # Lower speed threshold
        "--gap-hours", "4.0",      # Shorter gap for new tracks
    ]
    
    return run_command(cmd)

def example_production_run():
    """Example 4: Production run (processes all files)."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Production Run (All Files)")
    print("="*60)
    print("WARNING: This will process ALL files in your S3 bucket!")
    
    response = input("Do you want to proceed? (y/N): ")
    if response.lower() != 'y':
        print("Skipping production run.")
        return True
    
    cmd = [
        "python3", "s3_ais_processor.py",
        "--bucket", BUCKET_NAME,
        "--input-prefix", INPUT_PREFIX,
        "--output-prefix", OUTPUT_PREFIX,
        "--interpolate",
        "--interpolate-interval", "10",
    ]
    
    return run_command(cmd)

def main():
    """Run all examples."""
    print("S3 AIS Processing Pipeline Examples")
    print("="*60)
    
    # Check if we're in the right directory
    script_dir = Path(__file__).parent
    processor_script = script_dir / "s3_ais_processor.py"
    
    if not processor_script.exists():
        print(f"❌ Error: {processor_script} not found!")
        print(f"Make sure you're running this from the scripts directory.")
        sys.exit(1)
    
    # Change to scripts directory
    import os
    os.chdir(script_dir)
    
    examples = [
        ("Basic Processing", example_basic_processing),
        ("With Interpolation", example_with_interpolation),
        ("Custom Settings", example_custom_settings),
        ("Production Run", example_production_run),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nChoose an example to run:")
    print("  1-4: Run specific example")
    print("  all: Run examples 1-3 (skip production)")
    print("  q: Quit")
    
    choice = input("\nEnter your choice: ").strip().lower()
    
    if choice == 'q':
        print("Goodbye!")
        return
    
    success_count = 0
    total_count = 0
    
    if choice == 'all':
        # Run examples 1-3
        for name, func in examples[:-1]:  # Skip production run
            total_count += 1
            if func():
                success_count += 1
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        idx = int(choice) - 1
        name, func = examples[idx]
        total_count = 1
        if func():
            success_count = 1
    else:
        print("Invalid choice!")
        return
    
    print("\n" + "="*60)
    print(f"SUMMARY: {success_count}/{total_count} examples completed successfully")
    print("="*60)

if __name__ == "__main__":
    main()