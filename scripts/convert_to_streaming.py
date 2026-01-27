#!/usr/bin/env python3
"""
Convert materialized samples to smaller shards or MosaicML MDS format.

Two modes:
1. --format parquet: Split into ~64MB parquet shards (simple, works with PyArrow)
2. --format mds: Convert to MosaicML MDS format (best for distributed training)

Usage:
    # Split into smaller parquet files (~64MB each)
    python convert_to_streaming.py --format parquet --shard-size-mb 64

    # Convert to MDS format (requires: pip install mosaicml-streaming)
    python convert_to_streaming.py --format mds
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
WINDOW_SIZE = 928
NUM_FEATURES = 5
FLOATS_PER_SAMPLE = WINDOW_SIZE * NUM_FEATURES  # 4640
BYTES_PER_SAMPLE = FLOATS_PER_SAMPLE * 4  # 18560 bytes


def convert_to_small_parquet(
    bucket: str,
    input_prefix: str,
    output_prefix: str,
    shard_size_mb: int = 64,
    num_input_shards: int = 256,
):
    """Convert large parquet shards into smaller ~64MB shards."""

    target_bytes = shard_size_mb * 1024 * 1024
    samples_per_shard = target_bytes // BYTES_PER_SAMPLE

    logger.info(f"Target shard size: {shard_size_mb} MB")
    logger.info(f"Samples per output shard: {samples_per_shard:,}")

    output_shard_id = 0
    current_samples = []
    current_bytes = 0

    for input_shard_id in tqdm(range(num_input_shards), desc="Processing input shards"):
        input_path = f"s3://{bucket}/{input_prefix}samples_{input_shard_id:03d}.parquet"

        try:
            pf = pq.ParquetFile(input_path)
        except Exception as e:
            logger.error(f"Error opening {input_path}: {e}")
            continue

        # Stream through the input shard
        for batch in pf.iter_batches(batch_size=1000):
            features_list = batch['features'].to_pylist()

            for features in features_list:
                current_samples.append(features)
                current_bytes += BYTES_PER_SAMPLE

                # Write shard when it reaches target size
                if current_bytes >= target_bytes:
                    write_parquet_shard(
                        bucket, output_prefix, output_shard_id, current_samples
                    )
                    output_shard_id += 1
                    current_samples = []
                    current_bytes = 0

    # Write remaining samples
    if current_samples:
        write_parquet_shard(bucket, output_prefix, output_shard_id, current_samples)
        output_shard_id += 1

    logger.info(f"Created {output_shard_id} output shards")

    # Write index file
    write_index(bucket, output_prefix, output_shard_id, samples_per_shard)


def write_parquet_shard(bucket: str, prefix: str, shard_id: int, samples: list):
    """Write a single parquet shard."""
    features_flat = np.array(samples, dtype=np.float32)
    flat_array = pa.array(features_flat.ravel(), type=pa.float32())
    list_array = pa.FixedSizeListArray.from_arrays(flat_array, FLOATS_PER_SAMPLE)
    table = pa.table({'features': list_array})

    output_path = f"s3://{bucket}/{prefix}shard_{shard_id:05d}.parquet"
    pq.write_table(table, output_path, compression='zstd', compression_level=3)

    if shard_id % 100 == 0:
        logger.info(f"Written shard {shard_id}: {len(samples):,} samples")


def write_index(bucket: str, prefix: str, num_shards: int, samples_per_shard: int):
    """Write index file with shard metadata."""
    import json
    import boto3

    index = {
        "num_shards": num_shards,
        "samples_per_shard": samples_per_shard,
        "window_size": WINDOW_SIZE,
        "num_features": NUM_FEATURES,
        "feature_names": ["lat", "lon", "sog", "cog", "dt_seconds"],
    }

    s3 = boto3.client('s3')
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}index.json",
        Body=json.dumps(index, indent=2),
        ContentType='application/json'
    )
    logger.info(f"Written index.json")


def convert_to_mds(
    bucket: str,
    input_prefix: str,
    output_dir: str,
    num_input_shards: int = 256,
    compression: str = 'zstd',
):
    """Convert to MosaicML MDS format for optimal streaming."""
    import gc

    try:
        from streaming import MDSWriter
    except ImportError:
        logger.error("MosaicML streaming not installed. Run: pip install mosaicml-streaming")
        sys.exit(1)

    # MDS schema - store as raw bytes for efficiency
    columns = {
        'features': 'bytes',  # Raw float32 bytes (4640 * 4 = 18560 bytes)
    }

    logger.info(f"Converting to MDS format at {output_dir}")
    logger.info(f"Reading from s3://{bucket}/{input_prefix}")

    total_samples = 0

    with MDSWriter(
        out=output_dir,
        columns=columns,
        compression=compression,
        size_limit=67_108_864,  # 64 MB shards (MosaicML default)
    ) as writer:

        for input_shard_id in tqdm(range(num_input_shards), desc="Converting to MDS"):
            input_path = f"s3://{bucket}/{input_prefix}samples_{input_shard_id:03d}.parquet"

            try:
                pf = pq.ParquetFile(input_path)
            except Exception as e:
                logger.error(f"Error opening {input_path}: {e}")
                continue

            shard_samples = 0
            # Use smaller batches to reduce memory
            for batch in pf.iter_batches(batch_size=500):
                features_list = batch['features'].to_pylist()

                for features in features_list:
                    # Store as raw bytes
                    features_bytes = np.array(features, dtype=np.float32).tobytes()
                    writer.write({'features': features_bytes})
                    shard_samples += 1

                # Clear batch from memory
                del features_list
                del batch

            total_samples += shard_samples
            logger.info(f"Shard {input_shard_id}: {shard_samples:,} samples (total: {total_samples:,})")

            # Force garbage collection between shards
            gc.collect()

    logger.info(f"MDS conversion complete! Total samples: {total_samples:,}")


def main():
    parser = argparse.ArgumentParser(description="Convert materialized data to streaming format")
    parser.add_argument("--bucket", default="ais-pipeline-data-10179bbf-us-east-1")
    parser.add_argument("--input-prefix", default="materialized/")
    parser.add_argument("--output-prefix", default="streaming/")
    parser.add_argument("--format", choices=["parquet", "mds"], default="parquet",
                        help="Output format: parquet (small shards) or mds (MosaicML)")
    parser.add_argument("--shard-size-mb", type=int, default=64,
                        help="Target shard size in MB (parquet mode only)")
    parser.add_argument("--num-input-shards", type=int, default=256)
    parser.add_argument("--output-dir", default="./mds_output",
                        help="Local output directory (mds mode only)")

    args = parser.parse_args()

    if args.format == "parquet":
        convert_to_small_parquet(
            bucket=args.bucket,
            input_prefix=args.input_prefix,
            output_prefix=args.output_prefix,
            shard_size_mb=args.shard_size_mb,
            num_input_shards=args.num_input_shards,
        )
    else:
        convert_to_mds(
            bucket=args.bucket,
            input_prefix=args.input_prefix,
            output_dir=args.output_dir,
            num_input_shards=args.num_input_shards,
        )


if __name__ == "__main__":
    main()
