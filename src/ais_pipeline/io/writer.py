"""Writing Parquet output to S3."""
import io
import logging
from datetime import datetime
from typing import Optional, List

import boto3
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def write_parquet_to_s3(
    df: pl.DataFrame,
    bucket_name: str,
    s3_key: str,
    compression: str = "zstd",
    compression_level: int = 3,
    row_group_size: int = 100_000,
    s3_client=None,
) -> bool:
    """Write DataFrame to S3 as Parquet.

    Args:
        df: DataFrame to write
        bucket_name: S3 bucket name
        s3_key: S3 key for output file
        compression: Compression algorithm (zstd recommended)
        compression_level: Compression level
        row_group_size: Number of rows per row group
        s3_client: Optional boto3 S3 client

    Returns:
        True if successful
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    try:
        # Sort by track_id and timestamp for optimal read performance
        if "track_id" in df.columns and "timestamp" in df.columns:
            df = df.sort(["track_id", "timestamp"])

        # Convert to PyArrow table
        table = df.to_arrow()

        # Write to buffer
        buffer = io.BytesIO()
        pq.write_table(
            table,
            buffer,
            compression=compression,
            compression_level=compression_level,
            row_group_size=row_group_size,
        )

        # Upload to S3
        buffer.seek(0)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        file_size_mb = buffer.tell() / (1024 * 1024)
        logger.info(f"Wrote {df.height} rows ({file_size_mb:.1f} MB) to s3://{bucket_name}/{s3_key}")
        return True

    except Exception as e:
        logger.error(f"Error writing to S3: {e}")
        return False


def write_partitioned_parquet(
    df: pl.DataFrame,
    bucket_name: str,
    prefix: str,
    compression: str = "zstd",
    compression_level: int = 3,
    row_group_size: int = 100_000,
    s3_client=None,
) -> bool:
    """Write DataFrame to S3 partitioned by year/month/day.

    Args:
        df: DataFrame to write (must have timestamp column)
        bucket_name: S3 bucket name
        prefix: S3 prefix for output files
        compression: Compression algorithm
        compression_level: Compression level
        row_group_size: Number of rows per row group
        s3_client: Optional boto3 S3 client

    Returns:
        True if successful
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    if "timestamp" not in df.columns:
        logger.error("DataFrame must have timestamp column for partitioning")
        return False

    try:
        # Add partition columns
        df = df.with_columns([
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.day().alias("day"),
        ])

        # Group by partition columns
        partition_groups = df.group_by(["year", "month", "day"])

        success = True
        for partition_keys, partition_df in partition_groups:
            year, month, day = partition_keys

            # Create partition path
            partition_path = (
                f"{prefix.rstrip('/')}/"
                f"year={year}/"
                f"month={month:02d}/"
                f"day={day:02d}/"
                f"tracks.parquet"
            )

            # Drop partition columns before writing
            partition_df = partition_df.drop(["year", "month", "day"])

            if not write_parquet_to_s3(
                partition_df,
                bucket_name,
                partition_path,
                compression,
                compression_level,
                row_group_size,
                s3_client,
            ):
                success = False

        return success

    except Exception as e:
        logger.error(f"Error writing partitioned parquet: {e}")
        return False


def generate_track_catalog(
    df: pl.DataFrame,
    output_prefix: str,
) -> pl.DataFrame:
    """Generate track catalog for efficient lookup during training.

    Args:
        df: DataFrame with processed tracks
        output_prefix: Prefix where data files are stored

    Returns:
        Catalog DataFrame
    """
    if "track_id" not in df.columns or "timestamp" not in df.columns:
        logger.error("DataFrame must have track_id and timestamp columns")
        return pl.DataFrame()

    catalog = (
        df.group_by("track_id")
        .agg([
            pl.col("mmsi").first().alias("mmsi"),
            pl.col("timestamp").min().alias("start_time"),
            pl.col("timestamp").max().alias("end_time"),
            pl.count().alias("num_positions"),
        ])
        .with_columns([
            ((pl.col("end_time") - pl.col("start_time")).dt.total_seconds() / 3600)
            .alias("duration_hours")
        ])
    )

    return catalog


def write_track_catalog(
    catalog_df: pl.DataFrame,
    bucket_name: str,
    prefix: str,
    s3_client=None,
) -> bool:
    """Write track catalog to S3.

    Args:
        catalog_df: Catalog DataFrame
        bucket_name: S3 bucket name
        prefix: S3 prefix
        s3_client: Optional boto3 S3 client

    Returns:
        True if successful
    """
    catalog_key = f"{prefix.rstrip('/')}/track_catalog.parquet"
    return write_parquet_to_s3(
        catalog_df,
        bucket_name,
        catalog_key,
        s3_client=s3_client,
    )
