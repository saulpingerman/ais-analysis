#!/usr/bin/env python3
"""
Reorganize AIS data from day-partitioned to track-sharded format.

Batched approach: processes shards in groups to fit in memory.
With 400GB of data and 128GB RAM, we process ~50 shards at a time in 6 batches.

Input:  cleaned/year=YYYY/month=MM/day=DD/tracks.parquet (day-partitioned)
Output: sharded/shard=XXX/tracks.parquet (track-sharded, complete tracks per shard)
        sharded/catalog.parquet (track metadata with shard assignments)
"""

import argparse
import hashlib
import logging
import gc
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


def build_catalog_and_shard_map(
    bucket: str,
    input_files: list[str],
    num_shards: int,
    min_positions: int,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """
    First pass: build track catalog and compute shard assignments.
    Only keeps track metadata in memory, not full data.
    """
    logger.info("Pass 1: Building track catalog...")

    track_stats = {}

    for file_key in tqdm(input_files, desc="Scanning for catalog"):
        s3_path = f"s3://{bucket}/{file_key}"

        # Read only columns needed for catalog
        df = pl.read_parquet(s3_path, columns=["track_id", "timestamp", "mmsi"])

        # Aggregate per track
        file_stats = df.group_by("track_id").agg([
            pl.len().alias("count"),
            pl.col("timestamp").min().alias("min_ts"),
            pl.col("timestamp").max().alias("max_ts"),
            pl.col("mmsi").first().alias("mmsi"),
        ])

        for row in file_stats.iter_rows(named=True):
            tid = row["track_id"]
            if tid in track_stats:
                track_stats[tid]["num_positions"] += row["count"]
                track_stats[tid]["start_time"] = min(track_stats[tid]["start_time"], row["min_ts"])
                track_stats[tid]["end_time"] = max(track_stats[tid]["end_time"], row["max_ts"])
            else:
                track_stats[tid] = {
                    "track_id": tid,
                    "mmsi": row["mmsi"],
                    "num_positions": row["count"],
                    "start_time": row["min_ts"],
                    "end_time": row["max_ts"],
                }

        del df, file_stats

    # Build catalog DataFrame
    catalog = pl.DataFrame(list(track_stats.values()))

    # Filter by min positions
    catalog = catalog.filter(pl.col("num_positions") >= min_positions)

    # Compute shard IDs (vectorized would be better but this works)
    shard_ids = [get_shard_id(tid, num_shards) for tid in catalog["track_id"].to_list()]
    catalog = catalog.with_columns(pl.Series("shard_id", shard_ids).cast(pl.Int32))

    # Add duration
    catalog = catalog.with_columns(
        ((pl.col("end_time") - pl.col("start_time")).dt.total_seconds() / 3600).alias("duration_hours")
    )

    # Build track -> shard mapping
    track_to_shard = dict(zip(catalog["track_id"].to_list(), catalog["shard_id"].to_list()))

    logger.info(f"Catalog: {catalog.height:,} valid tracks")

    return catalog, track_to_shard


def process_shard_batch(
    bucket: str,
    input_files: list[str],
    output_prefix: str,
    shard_ids_in_batch: set[int],
    track_to_shard: dict[str, int],
    batch_num: int,
    total_batches: int,
) -> list[dict]:
    """
    Process a batch of shards: read all files, filter to batch's shards, write output.
    """
    logger.info(f"Batch {batch_num + 1}/{total_batches}: Processing shards {min(shard_ids_in_batch)}-{max(shard_ids_in_batch)}")

    # Accumulate data by shard
    shard_data = {sid: [] for sid in shard_ids_in_batch}
    valid_track_ids = set(track_to_shard.keys())

    for file_key in tqdm(input_files, desc=f"Batch {batch_num + 1} reading"):
        s3_path = f"s3://{bucket}/{file_key}"

        try:
            df = pl.read_parquet(s3_path)
        except Exception as e:
            logger.error(f"Error reading {file_key}: {e}")
            continue

        # Filter to valid tracks only
        df = df.filter(pl.col("track_id").is_in(valid_track_ids))

        if df.height == 0:
            del df
            continue

        # Add shard_id column
        df = df.with_columns(
            pl.col("track_id").replace(track_to_shard, default=None).cast(pl.Int32).alias("shard_id")
        )

        # Filter to shards in this batch
        df = df.filter(pl.col("shard_id").is_in(shard_ids_in_batch))

        if df.height == 0:
            del df
            continue

        # Distribute to shard accumulators (clone to avoid view references)
        for shard_id in shard_ids_in_batch:
            shard_df = df.filter(pl.col("shard_id") == shard_id).drop("shard_id").clone()
            if shard_df.height > 0:
                shard_data[shard_id].append(shard_df)
            del shard_df

        del df

    # Write shards
    shard_stats = []

    for shard_id in tqdm(sorted(shard_ids_in_batch), desc=f"Batch {batch_num + 1} writing"):
        if not shard_data[shard_id]:
            continue

        # Concatenate all data for this shard (diagonal handles schema differences)
        shard_df = pl.concat(shard_data[shard_id], how="diagonal")
        shard_data[shard_id] = None  # Free memory

        if shard_df.height == 0:
            del shard_df
            continue

        # Sort by track_id and timestamp
        shard_df = shard_df.sort(["track_id", "timestamp"])

        # Write to S3
        s3_path = f"s3://{bucket}/{output_prefix}shard={shard_id:03d}/tracks.parquet"
        shard_df.write_parquet(
            s3_path,
            compression="zstd",
            compression_level=3,
            row_group_size=100000,
        )

        shard_stats.append({
            "shard_id": shard_id,
            "num_tracks": shard_df["track_id"].n_unique(),
            "num_records": shard_df.height,
        })

        del shard_df

    # Clear the shard_data dict completely
    shard_data.clear()
    del shard_data

    # Force garbage collection
    gc.collect()

    return shard_stats


def shard_tracks_batched(
    bucket: str,
    input_prefix: str = "cleaned/",
    output_prefix: str = "sharded/",
    num_shards: int = 256,
    min_positions: int = 2,
    shards_per_batch: int = 50,
):
    """
    Reorganize day-partitioned data into track-sharded format.
    Uses batched approach to fit in memory.
    """
    s3_client = boto3.client('s3')

    # List all input files
    logger.info(f"Listing files in s3://{bucket}/{input_prefix}")
    input_files = list_parquet_files(s3_client, bucket, input_prefix)
    logger.info(f"Found {len(input_files)} input files")

    if not input_files:
        logger.error("No input files found")
        return

    # Pass 1: Build catalog and shard mapping
    catalog, track_to_shard = build_catalog_and_shard_map(
        bucket, input_files, num_shards, min_positions
    )

    # Check which shards already exist (for resume)
    existing_shards = set()
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{output_prefix}shard="):
        for obj in page.get('Contents', []):
            # Extract shard number from path like "sharded/shard=042/tracks.parquet"
            key = obj['Key']
            if '/shard=' in key and '/tracks.parquet' in key:
                try:
                    shard_str = key.split('/shard=')[1].split('/')[0]
                    existing_shards.add(int(shard_str))
                except (IndexError, ValueError):
                    pass

    if existing_shards:
        logger.info(f"Found {len(existing_shards)} existing shards, will skip them (resume mode)")

    # Determine batches (excluding existing shards)
    remaining_shards = sorted(set(range(num_shards)) - existing_shards)
    if not remaining_shards:
        logger.info("All shards already exist, nothing to do")
        return

    batches = [
        set(remaining_shards[i:i + shards_per_batch])
        for i in range(0, len(remaining_shards), shards_per_batch)
    ]

    logger.info(f"Processing {len(remaining_shards)} remaining shards in {len(batches)} batches of ~{shards_per_batch}")

    # Pass 2: Process each batch
    all_shard_stats = []
    for batch_num, shard_ids_in_batch in enumerate(batches):
        stats = process_shard_batch(
            bucket, input_files, output_prefix,
            shard_ids_in_batch, track_to_shard,
            batch_num, len(batches)
        )
        all_shard_stats.extend(stats)

        # Force memory release between batches
        gc.collect()
        logger.info(f"Batch {batch_num + 1} complete, memory released")

    # Write catalog
    logger.info("Writing track catalog...")
    catalog = catalog.select([
        "track_id", "mmsi", "shard_id", "num_positions",
        "start_time", "end_time", "duration_hours"
    ]).sort("track_id")

    catalog_path = f"s3://{bucket}/{output_prefix}catalog.parquet"
    catalog.write_parquet(catalog_path, compression="zstd")

    # Write shard index
    shard_index = pl.DataFrame(all_shard_stats)
    shard_index_path = f"s3://{bucket}/{output_prefix}shard_index.parquet"
    shard_index.write_parquet(shard_index_path, compression="zstd")

    # Summary
    total_tracks = catalog.height
    total_records = sum(s["num_records"] for s in all_shard_stats)

    logger.info("=" * 60)
    logger.info("SHARDING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total tracks: {total_tracks:,}")
    logger.info(f"  Total records: {total_records:,}")
    logger.info(f"  Shards created: {len(all_shard_stats)}")
    logger.info(f"  Avg tracks/shard: {total_tracks // max(len(all_shard_stats), 1):,}")
    logger.info(f"  Output: s3://{bucket}/{output_prefix}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Shard AIS data by track (batched)")
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
    parser.add_argument(
        "--shards-per-batch",
        type=int,
        default=50,
        help="Number of shards to process per batch"
    )

    args = parser.parse_args()

    shard_tracks_batched(
        bucket=args.bucket,
        input_prefix=args.input_prefix,
        output_prefix=args.output_prefix,
        num_shards=args.num_shards,
        min_positions=args.min_positions,
        shards_per_batch=args.shards_per_batch,
    )


if __name__ == "__main__":
    main()
