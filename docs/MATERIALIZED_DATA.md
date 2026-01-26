# Materialized Training Data

This document describes the pre-shuffled, materialized training data format optimized for ML training on AIS vessel trajectories.

## Overview

The materialized data contains **109 million training samples** pre-shuffled globally using the [Jane Street 2-pass algorithm](https://blog.janestreet.com/how-to-shuffle-a-big-dataset/). Each sample is a fixed-size window of 928 positions × 5 features, ready for direct use in sequence models.

**Key benefit:** Sequential reading through any shard gives you random samples from the ENTIRE dataset. No complex shuffling needed during training.

## Data Formats

### Original Format (Large Shards - 2.7 GB each)
```
s3://ais-pipeline-data-10179bbf-us-east-1/materialized/
├── samples_000.parquet   # ~426K samples, 2.7 GB compressed, 7.9 GB in memory!
├── ...
├── samples_255.parquet
└── validation.parquet
```

**WARNING:** These shards are too large for most GPU instances! Use the streaming format below.

### Streaming Format (Small Shards - 64 MB each) [RECOMMENDED]
```
s3://ais-pipeline-data-10179bbf-us-east-1/streaming/
├── shard_00000.parquet   # ~3,500 samples, 64 MB
├── shard_00001.parquet
├── ...
├── shard_NNNNN.parquet   # ~6,500 shards total
└── index.json
```

Convert using: `python scripts/convert_to_streaming.py --format parquet`

## Statistics

| Metric | Original | Streaming |
|--------|----------|-----------|
| Training samples | 109M | 109M |
| Shards | 256 | ~6,500 |
| Shard size (compressed) | 2.7 GB | 64 MB |
| **Shard size in memory** | **7.9 GB** | **~180 MB** |
| Safe for 32GB instance | NO | YES |

## Sample Format

Each parquet file has a single column `features` containing fixed-size arrays:

```
Schema:
  features: FixedSizeList[4640, float32]
```

Each row is a flattened window: `(928 positions × 5 features) = 4640 floats`

### Feature Order (per position)
| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | lat | -90 to 90 | Latitude in degrees |
| 1 | lon | -180 to 180 | Longitude in degrees |
| 2 | sog | 0 to ~50 | Speed over ground (knots) |
| 3 | cog | 0 to 360 | Course over ground (degrees) |
| 4 | dt_seconds | 0 to ~14400 | Time delta from previous position |

### Window Structure
```
Positions 0-127:   Input sequence (SEQ_LEN = 128)
Positions 128-927: Prediction horizon (MAX_HORIZON = 800)

Total: 928 positions per sample
```

## Loading Data (Memory-Efficient)

### CRITICAL: Do NOT load full shards into memory!

Each shard is **7.9 GB in memory**. On a 32GB instance, loading a full shard will OOM.

**Always use streaming with `iter_batches()`:**

### Basic Streaming (Recommended)
```python
import pyarrow.parquet as pq
import numpy as np

def stream_batches(shard_id: int, batch_size: int = 64):
    """Stream batches from a shard without loading it all into memory."""
    path = f"s3://ais-pipeline-data-10179bbf-us-east-1/materialized/samples_{shard_id:03d}.parquet"
    pf = pq.ParquetFile(path)

    for batch in pf.iter_batches(batch_size=batch_size):
        # Convert to numpy - only this batch is in memory
        features = np.array(batch['features'].to_pylist(), dtype=np.float32)
        features = features.reshape(-1, 928, 5)  # (batch_size, 928, 5)
        yield features

# Usage
for batch in stream_batches(shard_id=0, batch_size=64):
    input_seq = batch[:, :128, :]   # (64, 128, 5)
    target_seq = batch[:, 128:, :]  # (64, 800, 5)
    # Train on this batch
```

### PyTorch IterableDataset
```python
import torch
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
import numpy as np

class AISStreamingDataset(IterableDataset):
    """Memory-efficient streaming dataset for AIS trajectories."""

    def __init__(
        self,
        shard_ids: list[int],
        batch_size: int = 64,
        shuffle_shards: bool = True,
    ):
        self.shard_ids = shard_ids
        self.batch_size = batch_size
        self.shuffle_shards = shuffle_shards
        self.bucket = "ais-pipeline-data-10179bbf-us-east-1"

    def __iter__(self):
        # Shuffle shard order each epoch
        shard_order = list(self.shard_ids)
        if self.shuffle_shards:
            np.random.shuffle(shard_order)

        for shard_id in shard_order:
            path = f"s3://{self.bucket}/materialized/samples_{shard_id:03d}.parquet"
            pf = pq.ParquetFile(path)

            for batch in pf.iter_batches(batch_size=self.batch_size):
                features = np.array(batch['features'].to_pylist(), dtype=np.float32)
                features = features.reshape(-1, 928, 5)
                yield torch.from_numpy(features)

# Usage
train_shards = list(range(256))
dataset = AISStreamingDataset(train_shards, batch_size=64)
loader = DataLoader(dataset, batch_size=None)  # batch_size=None because we handle it

for batch in loader:
    # batch: (64, 928, 5)
    input_seq = batch[:, :128, :]
    target_seq = batch[:, 128:, :]
    # loss = model(input_seq, target_seq)
```

### Multi-Worker Loading
```python
class AISMultiWorkerDataset(IterableDataset):
    """Dataset that distributes shards across DataLoader workers."""

    def __init__(self, shard_ids: list[int], batch_size: int = 64):
        self.shard_ids = shard_ids
        self.batch_size = batch_size
        self.bucket = "ais-pipeline-data-10179bbf-us-east-1"

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            my_shards = self.shard_ids
        else:
            # Each worker gets a subset of shards
            my_shards = self.shard_ids[worker_info.id::worker_info.num_workers]

        np.random.shuffle(my_shards)

        for shard_id in my_shards:
            path = f"s3://{self.bucket}/materialized/samples_{shard_id:03d}.parquet"
            pf = pq.ParquetFile(path)

            for batch in pf.iter_batches(batch_size=self.batch_size):
                features = np.array(batch['features'].to_pylist(), dtype=np.float32)
                features = features.reshape(-1, 928, 5)
                yield torch.from_numpy(features)

# Usage with 4 workers
dataset = AISMultiWorkerDataset(list(range(256)), batch_size=64)
loader = DataLoader(dataset, batch_size=None, num_workers=4, prefetch_factor=2)
```

## Validation Data

```python
def stream_validation(batch_size: int = 64):
    """Stream validation batches."""
    path = "s3://ais-pipeline-data-10179bbf-us-east-1/materialized/validation.parquet"
    pf = pq.ParquetFile(path)

    for batch in pf.iter_batches(batch_size=batch_size):
        features = np.array(batch['features'].to_pylist(), dtype=np.float32)
        features = features.reshape(-1, 928, 5)
        yield features
```

## Feature Engineering

### Normalization
```python
def normalize_features(batch):
    """Normalize features to reasonable ranges."""
    # batch shape: (B, 928, 5)
    normalized = batch.clone()

    # lat: [-90, 90] -> [-1, 1]
    normalized[:, :, 0] = batch[:, :, 0] / 90.0

    # lon: [-180, 180] -> [-1, 1]
    normalized[:, :, 1] = batch[:, :, 1] / 180.0

    # sog: [0, ~50] -> [0, 1] (clip outliers)
    normalized[:, :, 2] = torch.clamp(batch[:, :, 2], 0, 30) / 30.0

    # cog: [0, 360] -> sin/cos encoding (handles wraparound)
    cog_rad = batch[:, :, 3] * (np.pi / 180.0)
    # Replace cog with sin and cos (increases feature dim by 1)

    # dt_seconds: log transform
    normalized[:, :, 4] = torch.log1p(batch[:, :, 4]) / 10.0  # log(14400) ≈ 9.6

    return normalized
```

### Course Encoding (Recommended)
```python
def encode_course(cog_degrees):
    """Encode course as sin/cos to handle 0°/360° wraparound."""
    radians = cog_degrees * (np.pi / 180.0)
    return np.sin(radians), np.cos(radians)
```

## Training Tasks

### Next Position Prediction
```python
# Input: positions 0-127
# Output: position 128 (lat, lon)
input_seq = batch[:, :128, :]      # (B, 128, 5)
target_pos = batch[:, 128, :2]     # (B, 2) - lat, lon
```

### Multi-Step Forecasting
```python
# Input: positions 0-127
# Output: positions 128-227 (next 100 positions)
input_seq = batch[:, :128, :]
target_seq = batch[:, 128:228, :2]  # (B, 100, 2)
```

### Full Horizon Prediction
```python
# Input: positions 0-127
# Output: positions 128-927 (full 800 position horizon)
input_seq = batch[:, :128, :]
target_seq = batch[:, 128:, :2]     # (B, 800, 2)
```

## Memory Budgeting

For a 32GB GPU instance:

| Component | Memory |
|-----------|--------|
| OS + Framework | ~4 GB |
| Model (typical) | 1-4 GB |
| Batch (64 samples) | 0.5 GB |
| Gradients + Optimizer | 2-8 GB |
| **Available headroom** | ~16-24 GB |

**Recommended batch sizes by instance:**
- 16 GB RAM: batch_size = 32
- 32 GB RAM: batch_size = 64-128
- 64 GB RAM: batch_size = 256

## Epochs and Iterations

```python
# Per epoch:
samples_per_epoch = 109_039_946
batch_size = 64
iterations_per_epoch = samples_per_epoch // batch_size  # ~1.7M iterations

# With 256 shards, each shard has ~426K samples
# At batch_size=64, that's ~6,650 batches per shard
```

## Why Pre-Shuffled?

Traditional approach:
1. Load data into buffer
2. Shuffle buffer
3. Train on buffer
4. Reload new data → **Model overfits to buffer before seeing diverse data**

Pre-shuffled approach:
1. Data already globally shuffled
2. Stream sequentially through any shard
3. Every batch is random from full dataset
4. **No overfitting to subsets**

## Regenerating the Data

If you need to regenerate the materialized data (e.g., different window size):

```bash
cd ~/ais-analysis
source ~/ais-env/bin/activate

python scripts/materialize_samples.py \
    --bucket ais-pipeline-data-10179bbf-us-east-1 \
    --input-prefix sharded/ \
    --output-prefix materialized/ \
    --num-output-shards 256 \
    --temp-dir /tmp/materialize_temp \
    --seed 42
```

**Requirements:** 128GB RAM instance, 2.5TB disk space for temp files.
