"""
Memory-efficient data loaders for AIS trajectory training.

Supports two formats:
1. Small parquet shards (~64MB each) - works with any setup
2. MosaicML MDS format - best for distributed training

Usage:
    from ais_pipeline.io.streaming_loader import AISStreamingDataset
    from torch.utils.data import DataLoader

    dataset = AISStreamingDataset(
        bucket="ais-pipeline-data-10179bbf-us-east-1",
        prefix="streaming/",  # or "materialized/" for old format
        batch_size=64,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=2)

    for batch in loader:
        # batch: (64, 928, 5)
        input_seq = batch[:, :128, :]
        target_seq = batch[:, 128:, :]
"""

import json
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

# Constants
WINDOW_SIZE = 928
NUM_FEATURES = 5
SEQ_LEN = 128
MAX_HORIZON = 800


class AISStreamingDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for AIS trajectories.

    Streams data directly from S3 without loading full shards into memory.
    Each shard is ~64MB, suitable for any GPU instance.
    """

    def __init__(
        self,
        bucket: str = "ais-pipeline-data-10179bbf-us-east-1",
        prefix: str = "streaming/",
        batch_size: int = 64,
        shard_ids: Optional[list[int]] = None,
        shuffle_shards: bool = True,
        is_validation: bool = False,
    ):
        """
        Args:
            bucket: S3 bucket name
            prefix: S3 prefix for data (e.g., "streaming/" or "materialized/")
            batch_size: Number of samples per batch
            shard_ids: List of shard IDs to use (None = auto-detect)
            shuffle_shards: Whether to shuffle shard order each epoch
            is_validation: If True, load validation.parquet instead
        """
        self.bucket = bucket
        self.prefix = prefix
        self.batch_size = batch_size
        self.shuffle_shards = shuffle_shards
        self.is_validation = is_validation

        if is_validation:
            self.shard_paths = [f"s3://{bucket}/{prefix}validation.parquet"]
        elif shard_ids is not None:
            self.shard_paths = [
                f"s3://{bucket}/{prefix}shard_{sid:05d}.parquet"
                for sid in shard_ids
            ]
        else:
            # Auto-detect shards
            self.shard_paths = self._discover_shards()

    def _discover_shards(self) -> list[str]:
        """Discover available shards from S3."""
        import boto3

        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')

        shard_paths = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.parquet') and 'shard_' in key:
                    shard_paths.append(f"s3://{self.bucket}/{key}")

        # Sort by shard number
        shard_paths.sort()
        return shard_paths

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate through batches, streaming from S3."""
        import pyarrow.parquet as pq

        # Handle multi-worker DataLoader
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Split shards among workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            my_shards = self.shard_paths[worker_id::num_workers]
        else:
            my_shards = self.shard_paths

        # Shuffle shard order
        if self.shuffle_shards and not self.is_validation:
            my_shards = list(my_shards)
            np.random.shuffle(my_shards)

        for shard_path in my_shards:
            try:
                pf = pq.ParquetFile(shard_path)
            except Exception as e:
                print(f"Warning: Could not open {shard_path}: {e}")
                continue

            for batch in pf.iter_batches(batch_size=self.batch_size):
                features = np.array(batch['features'].to_pylist(), dtype=np.float32)
                features = features.reshape(-1, WINDOW_SIZE, NUM_FEATURES)
                yield torch.from_numpy(features)


class AISMDSDataset:
    """
    MosaicML StreamingDataset wrapper for AIS data.

    Best for distributed, multi-node training.
    Requires: pip install mosaicml-streaming
    """

    def __init__(
        self,
        local: str,
        remote: Optional[str] = None,
        batch_size: int = 64,
        shuffle: bool = True,
    ):
        """
        Args:
            local: Local cache directory
            remote: Remote S3 path (e.g., "s3://bucket/mds/")
            batch_size: Batch size
            shuffle: Whether to shuffle
        """
        try:
            from streaming import StreamingDataset
        except ImportError:
            raise ImportError("Install mosaicml-streaming: pip install mosaicml-streaming")

        self.dataset = StreamingDataset(
            local=local,
            remote=remote,
            shuffle=shuffle,
            batch_size=batch_size,
        )
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for sample in self.dataset:
            # Decode bytes to numpy
            features = np.frombuffer(sample['features'], dtype=np.float32)
            features = features.reshape(WINDOW_SIZE, NUM_FEATURES)
            batch.append(features)

            if len(batch) == self.batch_size:
                yield torch.from_numpy(np.stack(batch))
                batch = []

        if batch:
            yield torch.from_numpy(np.stack(batch))

    def __len__(self):
        return len(self.dataset) // self.batch_size


def normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of trajectory windows.

    Args:
        batch: (B, 928, 5) tensor with [lat, lon, sog, cog, dt_seconds]

    Returns:
        Normalized tensor with same shape
    """
    normalized = batch.clone()

    # lat: [-90, 90] -> [-1, 1]
    normalized[:, :, 0] = batch[:, :, 0] / 90.0

    # lon: [-180, 180] -> [-1, 1]
    normalized[:, :, 1] = batch[:, :, 1] / 180.0

    # sog: clip to [0, 30] then scale to [0, 1]
    normalized[:, :, 2] = torch.clamp(batch[:, :, 2], 0, 30) / 30.0

    # cog: [0, 360] -> [-1, 1] via sin/cos would add dimension
    # For simplicity, just scale to [0, 1]
    normalized[:, :, 3] = batch[:, :, 3] / 360.0

    # dt_seconds: log transform, then scale
    normalized[:, :, 4] = torch.log1p(batch[:, :, 4]) / 10.0

    return normalized


def split_input_target(
    batch: torch.Tensor,
    seq_len: int = SEQ_LEN,
    target_len: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split batch into input sequence and target.

    Args:
        batch: (B, 928, 5) tensor
        seq_len: Input sequence length (default 128)
        target_len: Target length (default: all remaining = 800)

    Returns:
        input_seq: (B, seq_len, 5)
        target_seq: (B, target_len, 5) or (B, 800, 5)
    """
    input_seq = batch[:, :seq_len, :]

    if target_len is None:
        target_seq = batch[:, seq_len:, :]
    else:
        target_seq = batch[:, seq_len:seq_len + target_len, :]

    return input_seq, target_seq
