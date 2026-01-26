#!/usr/bin/env python3
"""
Materialize pre-shuffled training samples from track-sharded AIS data.

Uses Jane Street 2-pass shuffle algorithm:
1. Pass 1: Extract windows from tracks, randomly distribute to temp files
2. Pass 2: Shuffle each temp file, write to S3 as compressed parquet

Reference: https://blog.janestreet.com/how-to-shuffle-a-big-dataset/
"""

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
WINDOW_SIZE = 928  # SEQ_LEN (128) + MAX_HORIZON (800)
STRIDE = 32
NUM_FEATURES = 5  # lat, lon, sog, cog, dt_seconds
FEATURE_COLUMNS = ["lat", "lon", "sog", "cog", "dt_seconds"]
BYTES_PER_SAMPLE = WINDOW_SIZE * NUM_FEATURES * 4  # float32


def get_disk_free_gb(path: str = "/") -> float:
    """Get free disk space in GB."""
    stat = os.statvfs(path)
    return (stat.f_bavail * stat.f_frsize) / (1024 ** 3)


def check_disk_space(min_gb: float = 50.0, path: str = "/"):
    """Check if enough disk space is available, exit if not."""
    free_gb = get_disk_free_gb(path)
    if free_gb < min_gb:
        logger.error(f"DISK SPACE LOW: {free_gb:.1f} GB free, need {min_gb:.1f} GB minimum")
        logger.error("Stopping to prevent crash. Free up space or expand volume.")
        sys.exit(1)
    return free_gb


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint if exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {"pass1_completed_shards": [], "pass2_completed_piles": [], "phase": "pass1"}


def save_checkpoint(checkpoint_path: Path, checkpoint: dict):
    """Save checkpoint to disk."""
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f)


def extract_windows_from_track(features: np.ndarray) -> np.ndarray:
    """
    Extract sliding windows from a track's features.

    Args:
        features: (num_positions, NUM_FEATURES) array

    Returns:
        (num_windows, WINDOW_SIZE, NUM_FEATURES) array
    """
    num_positions = len(features)
    if num_positions < WINDOW_SIZE:
        return np.array([], dtype=np.float32).reshape(0, WINDOW_SIZE, NUM_FEATURES)

    num_windows = (num_positions - WINDOW_SIZE) // STRIDE + 1
    windows = np.zeros((num_windows, WINDOW_SIZE, NUM_FEATURES), dtype=np.float32)

    for i in range(num_windows):
        start = i * STRIDE
        windows[i] = features[start:start + WINDOW_SIZE]

    return windows


def pass1_extract_and_distribute(
    bucket: str,
    input_prefix: str,
    temp_dir: Path,
    num_output_shards: int,
    training_shard_ids: list[int],
    checkpoint: dict,
    checkpoint_path: Path,
):
    """
    Pass 1: Read input shards, extract windows, randomly distribute to temp files.
    """
    logger.info(f"Pass 1: Extracting windows from {len(training_shard_ids)} shards")

    # Open temp files for each output pile
    pile_files = {}
    pile_counts = {}
    for i in range(num_output_shards):
        pile_path = temp_dir / f"pile_{i:03d}.bin"
        # Append mode if resuming
        mode = 'ab' if checkpoint.get("pass1_completed_shards") else 'wb'
        pile_files[i] = open(pile_path, mode)
        pile_counts[i] = 0

    completed_shards = set(checkpoint.get("pass1_completed_shards", []))
    total_samples = 0
    total_tracks = 0

    try:
        for shard_id in tqdm(training_shard_ids, desc="Pass 1 (extract)"):
            if shard_id in completed_shards:
                logger.info(f"Skipping shard {shard_id} (already processed)")
                continue

            # Check disk space periodically
            if shard_id % 10 == 0:
                free_gb = check_disk_space(min_gb=50.0)
                logger.info(f"Disk space: {free_gb:.1f} GB free")

            # Load shard
            shard_path = f"s3://{bucket}/{input_prefix}shard={shard_id:03d}/tracks.parquet"
            try:
                df = pl.read_parquet(shard_path)
            except Exception as e:
                logger.error(f"Error reading shard {shard_id}: {e}")
                continue

            # Process each track
            shard_samples = 0
            shard_tracks = 0

            for track_df in df.partition_by("track_id"):
                # Extract features
                features = track_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
                features = np.nan_to_num(features, nan=0.0)

                if len(features) < WINDOW_SIZE:
                    continue

                # Extract windows
                windows = extract_windows_from_track(features)

                if len(windows) == 0:
                    continue

                shard_tracks += 1

                # Randomly assign each window to a pile
                pile_assignments = np.random.randint(0, num_output_shards, size=len(windows))

                for pile_id in range(num_output_shards):
                    mask = pile_assignments == pile_id
                    if not np.any(mask):
                        continue

                    pile_windows = windows[mask]
                    pile_files[pile_id].write(pile_windows.tobytes())
                    pile_counts[pile_id] += len(pile_windows)
                    shard_samples += len(pile_windows)

            total_samples += shard_samples
            total_tracks += shard_tracks

            # Free memory
            del df
            gc.collect()

            # Update checkpoint
            completed_shards.add(shard_id)
            checkpoint["pass1_completed_shards"] = list(completed_shards)
            save_checkpoint(checkpoint_path, checkpoint)

            logger.info(f"Shard {shard_id}: {shard_tracks:,} tracks, {shard_samples:,} samples")

    finally:
        # Close all pile files
        for f in pile_files.values():
            f.close()

    # Log pile statistics
    logger.info(f"Pass 1 complete: {total_tracks:,} tracks, {total_samples:,} samples")
    for pile_id, count in sorted(pile_counts.items()):
        if count > 0:
            logger.debug(f"  Pile {pile_id}: {count:,} samples")

    checkpoint["phase"] = "pass2"
    checkpoint["pile_counts"] = pile_counts
    save_checkpoint(checkpoint_path, checkpoint)

    return pile_counts


def pass2_shuffle_and_write(
    bucket: str,
    output_prefix: str,
    temp_dir: Path,
    num_output_shards: int,
    checkpoint: dict,
    checkpoint_path: Path,
    batch_size: int = 10000,
):
    """
    Pass 2: For each temp pile, shuffle and write to S3 as parquet.
    """
    logger.info(f"Pass 2: Shuffling and writing {num_output_shards} output shards")

    completed_piles = set(checkpoint.get("pass2_completed_piles", []))

    for pile_id in tqdm(range(num_output_shards), desc="Pass 2 (shuffle)"):
        if pile_id in completed_piles:
            logger.info(f"Skipping pile {pile_id} (already processed)")
            continue

        pile_path = temp_dir / f"pile_{pile_id:03d}.bin"

        if not pile_path.exists():
            logger.warning(f"Pile {pile_id} not found, skipping")
            continue

        file_size = pile_path.stat().st_size
        if file_size == 0:
            logger.warning(f"Pile {pile_id} is empty, skipping")
            pile_path.unlink()
            completed_piles.add(pile_id)
            continue

        num_samples = file_size // BYTES_PER_SAMPLE
        logger.info(f"Pile {pile_id}: {num_samples:,} samples ({file_size / 1e9:.2f} GB)")

        # Memory-map the file for efficient access
        data = np.memmap(pile_path, dtype=np.float32, mode='r')
        data = data.reshape(num_samples, WINDOW_SIZE, NUM_FEATURES)

        # Generate shuffled indices
        indices = np.random.permutation(num_samples)

        # Write to parquet in batches
        output_path = f"s3://{bucket}/{output_prefix}samples_{pile_id:03d}.parquet"

        # Collect all samples in shuffled order
        all_features_flat = []

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_indices = indices[batch_start:batch_end]

            # Read samples in shuffled order
            batch_data = np.array([data[i] for i in batch_indices])

            # Flatten for storage: (batch, 928, 5) -> (batch, 4640)
            batch_flat = batch_data.reshape(len(batch_data), -1)
            all_features_flat.append(batch_flat)

        # Concatenate all batches
        features_flat = np.vstack(all_features_flat)

        # Create PyArrow table with fixed-size list
        # Each row is a flat array of 4640 float32 values
        flat_array = pa.array(features_flat.ravel(), type=pa.float32())
        list_array = pa.FixedSizeListArray.from_arrays(flat_array, WINDOW_SIZE * NUM_FEATURES)

        table = pa.table({'features': list_array})

        # Write to S3
        pq.write_table(
            table,
            output_path,
            compression='zstd',
            compression_level=3,
        )

        logger.info(f"Written pile {pile_id} to {output_path}")

        # Clean up
        del data
        del features_flat
        del all_features_flat
        gc.collect()

        # Delete temp file to free disk space
        pile_path.unlink()
        logger.info(f"Deleted temp file for pile {pile_id}")

        # Update checkpoint
        completed_piles.add(pile_id)
        checkpoint["pass2_completed_piles"] = list(completed_piles)
        save_checkpoint(checkpoint_path, checkpoint)

    checkpoint["phase"] = "complete"
    save_checkpoint(checkpoint_path, checkpoint)
    logger.info("Pass 2 complete!")


def materialize_validation(
    bucket: str,
    input_prefix: str,
    output_prefix: str,
    validation_shard_id: int = 255,
):
    """
    Materialize validation shard separately into a single file.
    """
    logger.info(f"Materializing validation shard {validation_shard_id}")

    shard_path = f"s3://{bucket}/{input_prefix}shard={validation_shard_id:03d}/tracks.parquet"

    try:
        df = pl.read_parquet(shard_path)
    except Exception as e:
        logger.error(f"Error reading validation shard: {e}")
        return

    all_windows = []

    for track_df in tqdm(df.partition_by("track_id"), desc="Validation tracks"):
        features = track_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)

        if len(features) < WINDOW_SIZE:
            continue

        windows = extract_windows_from_track(features)
        if len(windows) > 0:
            all_windows.append(windows)

    if not all_windows:
        logger.warning("No validation windows extracted")
        return

    # Concatenate and shuffle
    all_windows = np.vstack(all_windows)
    np.random.shuffle(all_windows)

    logger.info(f"Validation samples: {len(all_windows):,}")

    # Flatten and write
    features_flat = all_windows.reshape(len(all_windows), -1)

    flat_array = pa.array(features_flat.ravel(), type=pa.float32())
    list_array = pa.FixedSizeListArray.from_arrays(flat_array, WINDOW_SIZE * NUM_FEATURES)

    table = pa.table({'features': list_array})

    output_path = f"s3://{bucket}/{output_prefix}validation.parquet"
    pq.write_table(
        table,
        output_path,
        compression='zstd',
        compression_level=3,
    )

    logger.info(f"Written validation to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Materialize pre-shuffled training samples")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--input-prefix", default="sharded/", help="Input prefix for sharded data")
    parser.add_argument("--output-prefix", default="materialized/", help="Output prefix")
    parser.add_argument("--num-output-shards", type=int, default=256, help="Number of output shards")
    parser.add_argument("--temp-dir", default="/home/ec2-user/materialize_temp", help="Temp directory for intermediate files")
    parser.add_argument("--validation-shard", type=int, default=255, help="Shard ID to use for validation")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation shard processing")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Setup temp directory
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = temp_dir / "checkpoint.json"

    # Load or initialize checkpoint
    if args.resume and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        logger.info(f"Resuming from checkpoint: phase={checkpoint['phase']}")
    else:
        checkpoint = {"pass1_completed_shards": [], "pass2_completed_piles": [], "phase": "pass1"}
        save_checkpoint(checkpoint_path, checkpoint)

    # Check initial disk space
    free_gb = check_disk_space(min_gb=100.0)
    logger.info(f"Initial disk space: {free_gb:.1f} GB free")

    # Training shard IDs (exclude validation)
    training_shard_ids = [i for i in range(256) if i != args.validation_shard]

    logger.info("=" * 60)
    logger.info("MATERIALIZE SAMPLES")
    logger.info("=" * 60)
    logger.info(f"  Bucket: {args.bucket}")
    logger.info(f"  Input: {args.input_prefix}")
    logger.info(f"  Output: {args.output_prefix}")
    logger.info(f"  Training shards: {len(training_shard_ids)}")
    logger.info(f"  Validation shard: {args.validation_shard}")
    logger.info(f"  Output shards: {args.num_output_shards}")
    logger.info(f"  Window size: {WINDOW_SIZE}")
    logger.info(f"  Stride: {STRIDE}")
    logger.info(f"  Temp dir: {temp_dir}")
    logger.info("=" * 60)

    # Pass 1: Extract and distribute
    if checkpoint["phase"] == "pass1":
        pass1_extract_and_distribute(
            bucket=args.bucket,
            input_prefix=args.input_prefix,
            temp_dir=temp_dir,
            num_output_shards=args.num_output_shards,
            training_shard_ids=training_shard_ids,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
        )

    # Pass 2: Shuffle and write
    if checkpoint["phase"] in ("pass1", "pass2"):
        pass2_shuffle_and_write(
            bucket=args.bucket,
            output_prefix=args.output_prefix,
            temp_dir=temp_dir,
            num_output_shards=args.num_output_shards,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
        )

    # Validation
    if not args.skip_validation:
        materialize_validation(
            bucket=args.bucket,
            input_prefix=args.input_prefix,
            output_prefix=args.output_prefix,
            validation_shard_id=args.validation_shard,
        )

    # Cleanup temp directory
    logger.info("Cleaning up temp directory...")
    shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("=" * 60)
    logger.info("MATERIALIZATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
