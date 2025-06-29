#!/usr/bin/env python3
"""
Read daily Parquet files and partition them by year, month, and day
using Hive partitioning scheme (year=YYYY/month=MM/day=DD).
"""
import pathlib
import shutil
import re
from tqdm import tqdm

SOURCE = pathlib.Path("/home/ec2-user/data/02_intermediate/parquet").expanduser()
DEST = pathlib.Path("/home/ec2-user/data/02_intermediate/partitioned_ais").expanduser()

def main():
    """
    Finds all daily parquet files, extracts the date,
    and copies them into a Hive-partitioned directory structure.
    """
    DEST.mkdir(parents=True, exist_ok=True)
    
    files_to_process = list(SOURCE.glob("*/*/*.parquet"))
    
    print(f"Found {len(files_to_process)} daily Parquet files to partition.")

    for f in tqdm(files_to_process, desc="Partitioning files"):
        # e.g., /path/to/2025/2025-01-01/2025-01-01.parquet
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(f))
        if not match:
            continue
            
        year, month, day = match.groups()
        
        partition_dir = DEST / f"year={year}" / f"month={month}" / f"day={day}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        dest_file = partition_dir / f.name
        
        if not dest_file.exists():
            shutil.copy(f, dest_file)

    print(f"✅  Partitioning complete. Data is in {DEST}")

if __name__ == "__main__":
    main() 