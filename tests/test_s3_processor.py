#!/usr/bin/env python3
"""
Test script for the S3 AIS processor.

This script runs basic tests to verify the S3 processor is working correctly.
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to the path for imports
script_dir = Path(__file__).parent.parent / "scripts"
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from s3_ais_processor import S3AISProcessor, load_config

def test_s3_connection():
    """Test S3 connection and bucket access."""
    print("Testing S3 connection...")
    
    try:
        config = load_config()
        bucket_name = "ais-pipeline-data-10179bbf-us-east-1"
        processor = S3AISProcessor(bucket_name, config)
        
        # Test listing files
        zip_files = processor.list_zip_files()
        print(f"✅ Found {len(zip_files)} ZIP files in S3 bucket")
        
        if zip_files:
            print("Sample files:")
            for i, file in enumerate(zip_files[:3]):
                print(f"  {i+1}. {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ S3 connection failed: {e}")
        return False

def test_processor_initialization():
    """Test processor initialization."""
    print("\nTesting processor initialization...")
    
    try:
        config = {
            'speed_thresh': 80.0,
            'gap_hours': 6.0,
            'chunk_size': 10000,
            'interpolate': False
        }
        
        processor = S3AISProcessor("ais-pipeline-data-10179bbf-us-east-1", config)
        
        print("✅ Processor initialized successfully")
        print(f"  Speed threshold: {processor.speed_thresh} knots")
        print(f"  Gap hours: {processor.gap_hours} hours")
        print(f"  Chunk size: {processor.chunk_size} rows")
        print(f"  Interpolation: {processor.interpolate}")
        
        return True
        
    except Exception as e:
        print(f"❌ Processor initialization failed: {e}")
        return False

def test_column_mapping():
    """Test column mapping functionality."""
    print("\nTesting column mapping...")
    
    try:
        config = {}
        processor = S3AISProcessor("ais-pipeline-data-10179bbf-us-east-1", config)
        
        # Test sample columns
        sample_columns = [
            "# Timestamp", "MMSI", "Latitude", "Longitude", "SOG", "COG", "Heading", "Ship Type"
        ]
        
        unique_cols = processor.make_unique_columns(sample_columns)
        rename_map = processor.map_columns(unique_cols)
        
        print("✅ Column mapping successful")
        print("  Rename map:", rename_map)
        
        return True
        
    except Exception as e:
        print(f"❌ Column mapping failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("S3 AIS Processor Tests")
    print("=" * 50)
    
    tests = [
        ("S3 Connection", test_s3_connection),
        ("Processor Initialization", test_processor_initialization),
        ("Column Mapping", test_column_mapping),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)