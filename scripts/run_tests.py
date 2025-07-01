#!/usr/bin/env python3
"""
Test Runner for AIS Pipeline

Simple script to run all tests in sequence.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run AIS pipeline tests")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--max-days", type=int, default=2, help="Max days for comprehensive test")
    
    args = parser.parse_args()
    
    print("AIS Pipeline Test Suite")
    print(f"S3 Bucket: {args.bucket}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    
    tests = []
    
    # Basic system test
    tests.append((
        f"python3 tests/test_s3_processor.py",
        "Basic System Validation"
    ))
    
    if not args.quick:
        # Comprehensive test
        tests.append((
            f"python3 tests/comprehensive_test.py --bucket {args.bucket} --max-days {args.max_days}",
            "Comprehensive Pipeline Test"
        ))
    
    # Run all tests
    passed = 0
    total = len(tests)
    
    for cmd, desc in tests:
        if run_command(cmd, desc):
            passed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()