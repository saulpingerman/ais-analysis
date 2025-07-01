#!/usr/bin/env python3
"""
Danish Maritime Authority AIS Data Downloader

Downloads AIS data from Danish Maritime Authority for specified date ranges,
checking S3 bucket first to avoid duplicate downloads. Handles both monthly
(pre-2024) and daily (2024+) data automatically.

Usage:
    python download_ais_data.py --start-date 2024-01-01 --end-date 2024-01-31
    python download_ais_data.py --start-date 2020-01-01 --end-date 2020-12-31
"""

import argparse
import logging
import re
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
import warnings

import boto3
import requests
from botocore.exceptions import ClientError, NoCredentialsError

warnings.filterwarnings("ignore")

# Add the scripts directory to path for imports
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

from s3_ais_processor import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AISDataDownloader:
    """Downloads AIS data from Danish Maritime Authority with S3 integration."""
    
    def __init__(self, bucket_name: str, s3_prefix: str = "data/01_raw/ais_dk/raw/"):
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix.rstrip('/') + '/'
        self.base_url = "http://web.ais.dk/aisdata/"
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client('s3')
            # Test connection
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"✅ Connected to S3 bucket: {bucket_name}")
        except NoCredentialsError:
            logger.error("❌ AWS credentials not found. Please configure AWS CLI.")
            sys.exit(1)
        except ClientError as e:
            logger.error(f"❌ Cannot access S3 bucket {bucket_name}: {e}")
            sys.exit(1)
    
    def get_existing_s3_files(self) -> set:
        """Get list of existing AIS files in S3 bucket."""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            existing_files = set()
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.s3_prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        filename = key.split('/')[-1]
                        if filename.startswith('aisdk-') and filename.endswith('.zip'):
                            existing_files.add(filename)
            
            logger.info(f"Found {len(existing_files)} existing AIS files in S3")
            return existing_files
            
        except Exception as e:
            logger.error(f"Error listing S3 files: {e}")
            return set()
    
    def generate_file_list(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Generate list of expected AIS filenames for date range."""
        files = []
        current_date = start_date
        
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            day = current_date.day
            
            # Determine if we expect daily or monthly files
            # Daily files started around March 2024 (update this based on actual data)
            # For now, let's be conservative and assume daily starting 2024-03
            if year >= 2024 and (year > 2024 or month >= 3):
                # Daily files
                filename = f"aisdk-{year:04d}-{month:02d}-{day:02d}.zip"
                files.append(filename)
                current_date += timedelta(days=1)
            else:
                # Monthly files
                filename = f"aisdk-{year:04d}-{month:02d}.zip"
                if filename not in files:  # Avoid duplicates when spanning months
                    files.append(filename)
                # Move to next month
                if month == 12:
                    current_date = datetime(year + 1, 1, 1)
                else:
                    current_date = datetime(year, month + 1, 1)
        
        return files
    
    def check_file_exists_remote(self, filename: str) -> bool:
        """Check if file exists on DMA server."""
        url = f"{self.base_url}{filename}"
        try:
            response = requests.head(url, timeout=30)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Could not check {filename}: {e}")
            return False
    
    def download_file(self, filename: str, temp_dir: Path) -> bool:
        """Download file from DMA to temporary directory."""
        url = f"{self.base_url}{filename}"
        local_path = temp_dir / filename
        
        try:
            logger.info(f"Downloading {filename}...")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Download with progress indication
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Simple progress indication
                        if total_size > 0 and downloaded % (1024 * 1024) == 0:  # Every MB
                            percent = (downloaded / total_size) * 100
                            if percent >= 10 and percent % 10 < 1:  # Show every 10%
                                logger.info(f"  Progress: {percent:.0f}%")
            
            file_size = local_path.stat().st_size
            logger.info(f"✅ Downloaded {filename} ({file_size / 1024 / 1024:.1f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {filename}: {e}")
            if local_path.exists():
                local_path.unlink()
            return False
    
    def upload_to_s3(self, local_path: Path, filename: str) -> bool:
        """Upload file to S3 bucket."""
        s3_key = f"{self.s3_prefix}{filename}"
        
        try:
            logger.info(f"Uploading {filename} to S3...")
            self.s3_client.upload_file(
                str(local_path),
                self.bucket_name,
                s3_key,
                ExtraArgs={'StorageClass': 'STANDARD_IA'}  # Use Infrequent Access for cost savings
            )
            logger.info(f"✅ Uploaded to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload {filename}: {e}")
            return False
    
    def download_date_range(self, start_date: datetime, end_date: datetime, 
                          force_redownload: bool = False) -> Tuple[int, int]:
        """
        Download AIS data for specified date range.
        
        Returns:
            Tuple of (successful_downloads, skipped_files)
        """
        logger.info(f"Processing AIS data from {start_date.date()} to {end_date.date()}")
        
        # Generate expected files
        expected_files = self.generate_file_list(start_date, end_date)
        logger.info(f"Expected {len(expected_files)} files for date range")
        
        # Get existing S3 files
        existing_s3_files = set() if force_redownload else self.get_existing_s3_files()
        
        # Filter out files that already exist
        files_to_download = []
        for filename in expected_files:
            if filename in existing_s3_files:
                logger.info(f"⏭️  Skipping {filename} (already exists in S3)")
            else:
                files_to_download.append(filename)
        
        if not files_to_download:
            logger.info("🎉 All files already exist in S3!")
            return 0, len(expected_files)
        
        logger.info(f"Need to download {len(files_to_download)} files")
        
        # Download missing files
        successful_downloads = 0
        skipped_files = len(expected_files) - len(files_to_download)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for filename in files_to_download:
                # Check if file exists on remote server
                if not self.check_file_exists_remote(filename):
                    logger.warning(f"⚠️  {filename} not found on server, skipping")
                    continue
                
                # Download file
                if self.download_file(filename, temp_path):
                    # Upload to S3
                    local_file = temp_path / filename
                    if self.upload_to_s3(local_file, filename):
                        successful_downloads += 1
                    
                    # Clean up local file
                    if local_file.exists():
                        local_file.unlink()
        
        logger.info(f"📊 Summary: {successful_downloads} downloaded, {skipped_files} skipped")
        return successful_downloads, skipped_files


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def main():
    parser = argparse.ArgumentParser(
        description="Download AIS data from Danish Maritime Authority",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data for January 2024 (daily files)
  python download_ais_data.py --start-date 2024-01-01 --end-date 2024-01-31
  
  # Download data for 2020 (monthly files)  
  python download_ais_data.py --start-date 2020-01-01 --end-date 2020-12-31
  
  # Download specific month
  python download_ais_data.py --start-date 2023-06-01 --end-date 2023-06-30
  
  # Force redownload existing files
  python download_ais_data.py --start-date 2024-01-01 --end-date 2024-01-01 --force
        """
    )
    
    parser.add_argument(
        "--start-date", 
        type=parse_date, 
        required=True,
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date", 
        type=parse_date, 
        required=True,
        help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--bucket", 
        help="S3 bucket name (overrides config)"
    )
    parser.add_argument(
        "--s3-prefix", 
        default="data/01_raw/ais_dk/raw/",
        help="S3 prefix for storing files"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force redownload even if files exist in S3"
    )
    parser.add_argument(
        "--config", 
        default="config.yaml",
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    # Validate date range
    if args.start_date > args.end_date:
        logger.error("Start date must be before or equal to end date")
        sys.exit(1)
    
    # Load bucket name from config or argument
    if args.bucket:
        bucket_name = args.bucket
    else:
        try:
            config = load_config(args.config)
            bucket_name = config.get('s3', {}).get('bucket_name')
            if not bucket_name:
                logger.error("No bucket name specified. Use --bucket or configure in config.yaml")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Could not load config: {e}")
            sys.exit(1)
    
    # Initialize downloader
    downloader = AISDataDownloader(bucket_name, args.s3_prefix)
    
    # Download data
    try:
        downloaded, skipped = downloader.download_date_range(
            args.start_date, 
            args.end_date,
            args.force
        )
        
        if downloaded > 0:
            logger.info(f"🎉 Successfully downloaded {downloaded} files to S3!")
        else:
            logger.info("ℹ️  No new files to download")
            
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()