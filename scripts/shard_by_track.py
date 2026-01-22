#!/usr/bin/env python3
"""
Reorganize AIS data from day-partitioned to track-sharded format.

Input:  cleaned/year=YYYY/month=MM/day=DD/tracks.parquet (day-partitioned)
Output: sharded/shard=XXX/tracks.parquet (track-sharded, complete tracks per shard)
        sharded/catalog.parquet (track metadata with shard assignments)

Each shard contains complete tracks - no track spans multiple shards.
Tracks are assigned to shards via hash(track_id) % num_shards.
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import boto3
import polars as pl
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_shard_id(track_id: str, num_shards: int) -> int:
    """Deterministically assign track to shard using hash."""
    hash_bytes = hashlib.md5(track_id.encode()).digest()
    hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
    return hash_int % num_shards


def list_parquet_files(s3_client, bucket: str, prefix: str) -> list[str]:
    """List all parquet files under a prefix."""
    paginator = s3_client.get_paginator('list_objects_v2')
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.parquet') and 'year=' in obj['Key']:
                files.append(obj['Key'])
    return sorted(files)


def shard_tracks(
    bucket: str,
    input_prefix: str = "cleaned/",
    output_prefix: str = "sharded/",
    num_shards: int = 256,
    min_positions: int = 2,
):
    """
    Reorganize day-partitioned data into track-sharded format.

    Args:
        bucket: S3 bucket name
        input_prefix: Prefix for day-partitioned data
        output_prefix: Prefix for sharded output
        num_shards: Number of shards to create
        min_positions: Minimum positions per track to include
    """
    s3_client = boto3.client('s3')

    # List all input files
    logger.info(f"Listing files in s3://{bucket}/{input_prefix}")
    input_files = list_parquet_files(s3_client, bucket, input_prefix)
    logger.info(f"Found {len(input_files)} input files")

    if not input_files:
        logger.error("No input files found")
        return

    # Initialize shard buffers (track_id -> list of dataframes)
    # We'll accumulate data per track, then assign to shards
    logger.info("Reading all data and grouping by track...")

    # Read all files and concatenate
    all_dfs = []
    for file_key in tqdm(input_files, desc="Reading files"):
        s3_path = f"s3://{bucket}/{file_key}"
        df = pl.read_parquet(s3_path)
        all_dfs.append(df)

    logger.info("Concatenating all data...")
    full_df = pl.concat(all_dfs)
    del all_dfs  # Free memory

    logger.info(f"Total records: {full_df.height:,}")

    # Get unique tracks and their stats
    logger.info("Computing track statistics...")
    track_stats = full_df.group_by("track_id").agg([
        pl.count().alias("num_positions"),
        pl.col("timestamp").min().alias("start_time"),
        pl.col("timestamp").max().alias("end_time"),
        pl.col("mmsi").first().alias("mmsi"),
    ])

    # Filter tracks by minimum positions
    valid_tracks = track_stats.filter(pl.col("num_positions") >= min_positions)
    logger.info(f"Tracks with >= {min_positions} positions: {valid_tracks.height:,} / {track_stats.height:,}")

    # Assign shard IDs
    logger.info("Assigning tracks to shards...")
    valid_tracks = valid_tracks.with_columns(
        pl.col("track_id").map_elements(
            lambda tid: get_shard_id(tid, num_shards),
            return_dtype=pl.Int32
        ).alias("shard_id")
    )

    # Add duration
    valid_tracks = valid_tracks.with_columns(
        ((pl.col("end_time") - pl.col("start_time")).dt.total_seconds() / 3600).alias("duration_hours")
    )

    # Create track_id to shard_id mapping
    track_to_shard = dict(zip(
        valid_tracks["track_id"].to_list(),
        valid_tracks["shard_id"].to_list()
    ))
    valid_track_ids = set(track_to_shard.keys())

    # Filter main data to only valid tracks
    logger.info("Filtering data to valid tracks...")
    full_df = full_df.filter(pl.col("track_id").is_in(valid_track_ids))
    logger.info(f"Records after filtering: {full_df.height:,}")

    # Add shard_id to main data
    full_df = full_df.with_columns(
        pl.col("track_id").map_elements(
            lambda tid: track_to_shard.get(tid, -1),
            return_dtype=pl.Int32
        ).alias("shard_id")
    )

    # Write shards
    logger.info(f"Writing {num_shards} shards to s3://{bucket}/{output_prefix}")

    shard_stats = []
    for shard_id in tqdm(range(num_shards), desc="Writing shards"):
        shard_df = full_df.filter(pl.col("shard_id") == shard_id)

        if shard_df.height == 0:
            continue

        # Remove shard_id column before writing
        shard_df = shard_df.drop("shard_id")

        # Sort by track_id and timestamp for efficient reading
        shard_df = shard_df.sort(["track_id", "timestamp"])

        # Write to S3
        s3_path = f"s3://{bucket}/{output_prefix}shard={shard_id:03d}/tracks.parquet"
        shard_df.write_parquet(
            s3_path,
            compression="zstd",
            compression_level=3,
            row_group_size=100000,
        )

        num_tracks = shard_df["track_id"].n_unique()
        shard_stats.append({
            "shard_id": shard_id,
            "num_tracks": num_tracks,
            "num_records": shard_df.height,
        })

    # Write catalog
    logger.info("Writing track catalog...")
    catalog = valid_tracks.select([
        "track_id",
        "mmsi",
        "shard_id",
        "num_positions",
        "start_time",
        "end_time",
        "duration_hours",
    ]).sort("track_id")

    catalog_path = f"s3://{bucket}/{output_prefix}catalog.parquet"
    catalog.write_parquet(catalog_path, compression="zstd")

    # Write shard index
    shard_index = pl.DataFrame(shard_stats)
    shard_index_path = f"s3://{bucket}/{output_prefix}shard_index.parquet"
    shard_index.write_parquet(shard_index_path, compression="zstd")

    # Summary
    logger.info("=" * 60)
    logger.info("SHARDING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total tracks: {valid_tracks.height:,}")
    logger.info(f"  Total records: {full_df.height:,}")
    logger.info(f"  Shards created: {len(shard_stats)}")
    logger.info(f"  Avg tracks/shard: {valid_tracks.height // num_shards:,}")
    logger.info(f"  Output: s3://{bucket}/{output_prefix}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Shard AIS data by track")
    parser.add_argument(
        "--bucket",
        default="ais-pipeline-data-10179bbf-us-east-1",
        help="S3 bucket name"
    )
    parser.add_argument(
        "--input-prefix",
        default="cleaned/",
        help="Input prefix for day-partitioned data"
    )
    parser.add_argument(
        "--output-prefix",
        default="sharded/",
        help="Output prefix for sharded data"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=256,
        help="Number of shards to create"
    )
    parser.add_argument(
        "--min-positions",
        type=int,
        default=2,
        help="Minimum positions per track"
    )

    args = parser.parse_args()

    shard_tracks(
        bucket=args.bucket,
        input_prefix=args.input_prefix,
        output_prefix=args.output_prefix,
        num_shards=args.num_shards,
        min_positions=args.min_positions,
    )


if __name__ == "__main__":
    main()
