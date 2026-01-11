"""Reading ZIP/CSV files from S3."""
import io
import re
import logging
import zipfile
from collections import Counter
from typing import Dict, List, Optional

import boto3
import polars as pl
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Column mapping patterns for AIS data
COLUMN_PATTERNS = {
    "timestamp": re.compile(r"^#?\s*timestamp$", re.I),
    "mmsi": re.compile(r"^mmsi$", re.I),
    "lat": re.compile(r"^lat(itude)?$", re.I),
    "lon": re.compile(r"^lon(gitude)?$", re.I),
    "sog": re.compile(r"^sog$", re.I),
    "cog": re.compile(r"^cog$", re.I),
    "heading": re.compile(r"^heading$", re.I),
    "ship_type": re.compile(r"^ship.?type$", re.I),
    "imo": re.compile(r"^imo$", re.I),
    "name": re.compile(r"^name$", re.I),
    "callsign": re.compile(r"^callsign$", re.I),
}

# Timestamp formats to try
TIMESTAMP_FORMATS = [
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
]


def map_columns(source_columns: List[str]) -> Dict[str, str]:
    """Map source columns to canonical names.

    Args:
        source_columns: List of column names from source file

    Returns:
        Dictionary mapping source column names to canonical names
    """
    rename_map = {}
    for canonical, pattern in COLUMN_PATTERNS.items():
        for source_col in source_columns:
            if pattern.match(source_col):
                if source_col not in rename_map:
                    rename_map[source_col] = canonical
                    break
    return rename_map


def list_raw_files(
    bucket_name: str,
    prefix: str = "raw/",
    s3_client=None,
) -> List[str]:
    """List all ZIP files in S3 bucket.

    Args:
        bucket_name: S3 bucket name
        prefix: S3 prefix for raw files
        s3_client: Optional boto3 S3 client

    Returns:
        Sorted list of S3 keys for ZIP files
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        zip_files = []

        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if key.lower().endswith(".zip") and "aisdk-" in key:
                        zip_files.append(key)

        return sorted(zip_files)
    except ClientError as e:
        logger.error(f"Error listing S3 objects: {e}")
        return []


def read_zip_from_s3(
    bucket_name: str,
    s3_key: str,
    s3_client=None,
) -> Optional[pl.DataFrame]:
    """Read and process a ZIP file from S3.

    Args:
        bucket_name: S3 bucket name
        s3_key: S3 key for the ZIP file
        s3_client: Optional boto3 S3 client

    Returns:
        Combined DataFrame from all CSVs in the ZIP, or None on error
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    try:
        logger.info(f"Reading {s3_key}")

        # Download ZIP file to memory
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        zip_data = response["Body"].read()

        all_data = []

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            csv_members = [m for m in zf.infolist() if m.filename.lower().endswith(".csv")]

            for member in csv_members:
                try:
                    df = read_csv_from_zip(zf, member.filename)
                    if df is not None and not df.is_empty():
                        all_data.append(df)
                except Exception as e:
                    logger.warning(f"Error processing CSV {member.filename}: {e}")
                    continue

        if not all_data:
            logger.warning(f"No valid data found in {s3_key}")
            return None

        # Combine all data
        combined_df = pl.concat(all_data, how="diagonal")

        # Ensure we have lat/lon columns (not latitude/longitude)
        if "latitude" in combined_df.columns:
            combined_df = combined_df.rename({"latitude": "lat"})
        if "longitude" in combined_df.columns:
            combined_df = combined_df.rename({"longitude": "lon"})

        logger.info(f"Read {combined_df.height} records from {s3_key}")
        return combined_df

    except Exception as e:
        logger.error(f"Error reading {s3_key}: {e}")
        return None


def read_csv_from_zip(zf: zipfile.ZipFile, filename: str) -> Optional[pl.DataFrame]:
    """Read a single CSV file from a ZIP archive.

    Args:
        zf: ZipFile object
        filename: Name of the CSV file within the ZIP

    Returns:
        DataFrame or None on error
    """
    with zf.open(filename) as csv_stream:
        text_stream = io.TextIOWrapper(csv_stream, encoding="utf-8", errors="ignore")

        try:
            df = pl.read_csv(
                text_stream,
                separator=",",
                ignore_errors=True,
                truncate_ragged_lines=True,
                infer_schema_length=1000,
            )

            # Map columns to canonical names
            rename_map = map_columns(df.columns)
            if not rename_map:
                return None

            # Select and rename columns
            cols_to_keep = [col for col in rename_map.keys() if col in df.columns]
            if not cols_to_keep:
                return None

            df = df.select(cols_to_keep).rename(rename_map)

            # Parse timestamp
            if "timestamp" in df.columns:
                df = parse_timestamp(df)
                if df is None:
                    return None

            return df

        except Exception as e:
            logger.warning(f"Error reading CSV {filename}: {e}")
            return None


def parse_timestamp(df: pl.DataFrame) -> Optional[pl.DataFrame]:
    """Parse timestamp column trying multiple formats.

    Args:
        df: DataFrame with timestamp column

    Returns:
        DataFrame with parsed timestamp or None if parsing fails
    """
    for fmt in TIMESTAMP_FORMATS:
        try:
            parsed_df = df.with_columns([
                pl.col("timestamp").str.strptime(
                    pl.Datetime,
                    format=fmt,
                    strict=False
                )
            ]).filter(pl.col("timestamp").is_not_null())

            if not parsed_df.is_empty():
                return parsed_df
        except Exception:
            continue

    return None
