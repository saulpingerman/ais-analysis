#!/usr/bin/env python3
"""
Parallel MDS conversion - runs multiple workers to speed up conversion.

Each worker handles a subset of input shards and writes to its own output directory.
MosaicML StreamingDataset can read from multiple directories.

Usage:
    python parallel_mds_convert.py --num-workers 8
"""

import argparse
import gc
import logging
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
WINDOW_SIZE = 928
NUM_FEATURES = 5
FLOATS_PER_SAMPLE = WINDOW_SIZE * NUM_FEATURES


def convert_worker(
    worker_id: int,
    input_shard_ids: list[int],
    bucket: str,
    input_prefix: str,
    output_base: str,
):
    """Worker function to convert a subset of shards to MDS."""
    from streaming import MDSWriter

    output_dir = f"{output_base}/worker_{worker_id:02d}"

    columns = {
        'features': 'bytes',
    }

    total_samples = 0

    logger.info(f"Worker {worker_id}: Converting shards {input_shard_ids[0]}-{input_shard_ids[-1]} to {output_dir}")

    with MDSWriter(
        out=output_dir,
        columns=columns,
        compression='zstd',
        size_limit=67_108_864,
    ) as writer:

        for input_shard_id in input_shard_ids:
            input_path = f"s3://{bucket}/{input_prefix}samples_{input_shard_id:03d}.parquet"

            try:
                pf = pq.ParquetFile(input_path)
            except Exception as e:
                logger.error(f"Worker {worker_id}: Error opening {input_path}: {e}")
                continue

            shard_samples = 0
            for batch in pf.iter_batches(batch_size=500):
                features_list = batch['features'].to_pylist()

                for features in features_list:
                    features_bytes = np.array(features, dtype=np.float32).tobytes()
                    writer.write({'features': features_bytes})
                    shard_samples += 1

                del features_list
                del batch

            total_samples += shard_samples
            logger.info(f"Worker {worker_id}: Shard {input_shard_id} done ({shard_samples:,} samples, total: {total_samples:,})")
            gc.collect()

    logger.info(f"Worker {worker_id}: Complete! {total_samples:,} total samples")
    return worker_id, total_samples


def main():
    parser = argparse.ArgumentParser(description="Parallel MDS conversion")
    parser.add_argument("--bucket", default="ais-pipeline-data-10179bbf-us-east-1")
    parser.add_argument("--input-prefix", default="materialized/")
    parser.add_argument("--output-base", default="s3://ais-pipeline-data-10179bbf-us-east-1/mds")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-input-shards", type=int, default=256)

    args = parser.parse_args()

    # Distribute input shards across workers
    all_shards = list(range(args.num_input_shards))
    shards_per_worker = len(all_shards) // args.num_workers

    worker_shards = []
    for i in range(args.num_workers):
        start = i * shards_per_worker
        if i == args.num_workers - 1:
            # Last worker gets any remaining shards
            end = len(all_shards)
        else:
            end = start + shards_per_worker
        worker_shards.append(all_shards[start:end])

    logger.info(f"Starting {args.num_workers} workers")
    logger.info(f"Each worker handles ~{shards_per_worker} input shards")
    logger.info(f"Output: {args.output_base}/worker_XX/")

    start_time = time.time()

    # Run workers in parallel using multiprocessing
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for worker_id, shard_list in enumerate(worker_shards):
            future = executor.submit(
                convert_worker,
                worker_id,
                shard_list,
                args.bucket,
                args.input_prefix,
                args.output_base,
            )
            futures.append(future)

        # Wait for all workers to complete
        total_samples = 0
        for future in as_completed(futures):
            try:
                worker_id, samples = future.result()
                total_samples += samples
                logger.info(f"Worker {worker_id} finished with {samples:,} samples")
            except Exception as e:
                logger.error(f"Worker failed: {e}")

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PARALLEL CONVERSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples: {total_samples:,}")
    logger.info(f"Time: {elapsed/3600:.1f} hours")
    logger.info(f"Output directories: {args.output_base}/worker_00/ through worker_{args.num_workers-1:02d}/")
    logger.info("=" * 60)

    # Write a manifest file listing all worker directories
    import boto3
    import json

    manifest = {
        "num_workers": args.num_workers,
        "total_samples": total_samples,
        "worker_dirs": [f"{args.output_base}/worker_{i:02d}" for i in range(args.num_workers)],
        "window_size": WINDOW_SIZE,
        "num_features": NUM_FEATURES,
        "feature_names": ["lat", "lon", "sog", "cog", "dt_seconds"],
    }

    # Extract bucket and key from output_base
    if args.output_base.startswith("s3://"):
        parts = args.output_base[5:].split("/", 1)
        manifest_bucket = parts[0]
        manifest_key = parts[1] + "/manifest.json" if len(parts) > 1 else "manifest.json"

        s3 = boto3.client('s3')
        s3.put_object(
            Bucket=manifest_bucket,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2),
            ContentType='application/json'
        )
        logger.info(f"Written manifest to s3://{manifest_bucket}/{manifest_key}")


if __name__ == "__main__":
    main()
