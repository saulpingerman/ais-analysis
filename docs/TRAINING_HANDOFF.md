# Training Handoff Document

## Summary

AIS vessel trajectory data has been fully processed and is ready for ML training. The data is stored in MosaicML MDS format, optimized for memory-efficient streaming on GPU instances.

## Data Location

```
s3://ais-pipeline-data-10179bbf-us-east-1/mds/
```

## Key Statistics

| Metric | Value |
|--------|-------|
| Training samples | 108,726,984 |
| Validation samples | ~313,000 |
| Total size | 632 GB |
| Shard size | ~22 MB (zstd compressed) |
| Number of shards | ~30,000 |
| Window size | 928 positions × 5 features |

## Sample Structure

Each sample is a trajectory window:
- **Positions 0-127**: Input sequence (SEQ_LEN = 128)
- **Positions 128-927**: Prediction horizon (MAX_HORIZON = 800)

Features per position:
| Index | Feature | Range |
|-------|---------|-------|
| 0 | lat | -90 to 90 |
| 1 | lon | -180 to 180 |
| 2 | sog | 0 to ~50 knots |
| 3 | cog | 0 to 360 degrees |
| 4 | dt_seconds | 0 to ~14400 |

## Quick Start

### Installation

```bash
pip install mosaicml-streaming torch numpy
```

### Basic Usage

```python
from streaming import StreamingDataset
import numpy as np

dataset = StreamingDataset(
    remote='s3://ais-pipeline-data-10179bbf-us-east-1/mds',
    local='/tmp/mds-cache',
    shuffle=True,
)

for sample in dataset:
    features = np.frombuffer(sample['features'], dtype=np.float32)
    features = features.reshape(928, 5)

    input_seq = features[:128]    # (128, 5)
    target_seq = features[128:]   # (800, 5)
```

### Using the Wrapper

```python
from ais_pipeline.io.streaming_loader import AISMDSDataset, normalize_batch, split_input_target

dataset = AISMDSDataset(
    remote='s3://ais-pipeline-data-10179bbf-us-east-1/mds',
    local='/tmp/mds-cache',
    batch_size=64,
)

for batch in dataset:
    # batch: (64, 928, 5) torch.Tensor
    batch = normalize_batch(batch)
    input_seq, target_seq = split_input_target(batch)
    # input_seq: (64, 128, 5)
    # target_seq: (64, 800, 5)
```

## Validation Data

Validation data is in parquet format:

```python
import pyarrow.parquet as pq
import numpy as np

pf = pq.ParquetFile('s3://ais-pipeline-data-10179bbf-us-east-1/materialized/validation.parquet')
for batch in pf.iter_batches(batch_size=64):
    features = np.array(batch['features'].to_pylist(), dtype=np.float32)
    features = features.reshape(-1, 928, 5)
```

## Recommended Batch Sizes

| GPU Memory | Batch Size |
|------------|------------|
| 16 GB | 32 |
| 32 GB (g6e.xlarge) | 64-128 |
| 64 GB | 256 |

## Instance Recommendations

- **Training**: g6e.xlarge (32 GB GPU, NVIDIA L4) or larger
- **Data processing**: r6i.4xlarge (128 GB RAM, 16 vCPU)

## Files Reference

- Data loader: `src/ais_pipeline/io/streaming_loader.py`
- Documentation: `docs/MATERIALIZED_DATA.md`
- MDS conversion script: `scripts/parallel_mds_convert.py`
- Materialization script: `scripts/materialize_samples.py`

## Data Pipeline Overview

```
Raw AIS Data (6.9B records, 363 days)
         ↓
    Processing & Filtering
         ↓
Track-Sharded Parquet (651K tracks, 44 GB)
         ↓
    Window Extraction (stride=32)
         ↓
Materialized Parquet (109M samples, 645 GB)
         ↓
    MDS Conversion
         ↓
MDS Format (109M samples, 632 GB) ← YOU ARE HERE
```

## Notes

- Data is globally pre-shuffled using the Jane Street 2-pass algorithm
- Sequential reading gives random samples from the entire dataset
- MDS format supports automatic distributed training (multi-GPU, multi-node)
- Local cache (`/tmp/mds-cache`) is populated on first access; subsequent reads are fast
