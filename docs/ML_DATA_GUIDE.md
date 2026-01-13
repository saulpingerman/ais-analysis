# AIS Data Guide for ML Training

This document provides everything needed to work with the processed AIS vessel tracking data for machine learning applications.

---

## 1. Data Location and Access

### S3 Bucket Structure
```
s3://your-ais-bucket/
├── cleaned/
│   ├── year=2024/month=12/day=01/tracks.parquet
│   ├── year=2024/month=12/day=02/tracks.parquet
│   ├── ... (Dec 2024 + Jan 2025 + Feb 1-7 2025)
│   ├── year=2025/month=02/day=07/tracks.parquet
│   └── track_catalog.parquet
└── state/
    ├── track_continuity.json
    └── processing_checkpoint.json
```

### FSx for Lustre Setup
When mounting FSx for Lustre linked to this S3 bucket:
- Mount point: `/mnt/fsx/`
- Data appears at: `/mnt/fsx/cleaned/year=YYYY/month=MM/day=DD/tracks.parquet`
- First read of each file triggers lazy load from S3
- Subsequent reads are at full Lustre speed (GB/s)

### Direct S3 Access (if not using FSx)
```python
import polars as pl
import s3fs

fs = s3fs.S3FileSystem()
df = pl.read_parquet("s3://your-ais-bucket/cleaned/year=2025/month=02/day=01/tracks.parquet")
```

---

## 2. Data Schema

### Main Track Files (`tracks.parquet`)

| Column | Type | Description | ML Notes |
|--------|------|-------------|----------|
| `timestamp` | datetime[μs] | Position timestamp (UTC) | Primary temporal feature |
| `mmsi` | int64 | Maritime Mobile Service Identity (vessel ID) | Use for grouping, not as feature |
| `track_id` | string | Unique track identifier | **Primary key for sequences** |
| `lat` | float64 | Latitude in decimal degrees | Normalize to [-1, 1] or use as-is |
| `lon` | float64 | Longitude in decimal degrees | Normalize to [-1, 1] or use as-is |
| `sog` | float64 | Speed over ground (knots) | Key velocity feature |
| `cog` | float64 | Course over ground (degrees, 0-360) | Consider sin/cos encoding |
| `heading` | int64 | Vessel heading (degrees), may be null | Optional feature |
| `ship_type` | string | Vessel type code | Categorical feature |
| `dt_seconds` | float32 | Time delta from previous position (seconds) | **Critical for sequence models** |
| `cluster_assignment` | string | "A" or "B" for collision-split tracks, null otherwise | Filter these if needed |
| `imo` | string | IMO number (may be null) | Vessel metadata |
| `name` | string | Vessel name (may be null) | Metadata only |
| `callsign` | string | Radio callsign (may be null) | Metadata only |

### Track Catalog (`track_catalog.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `track_id` | string | Unique track identifier |
| `mmsi` | int64 | Vessel MMSI |
| `start_time` | datetime | First position timestamp |
| `end_time` | datetime | Last position timestamp |
| `num_positions` | uint32 | Number of positions in track |
| `duration_hours` | float64 | Track duration in hours |

**Use the catalog for:**
- Filtering tracks by minimum length (e.g., `num_positions >= 100`)
- Sampling tracks for training batches
- Stratified sampling by duration or vessel type

---

## 3. Track Structure

### What is a Track?
A **track** is a continuous sequence of positions from a single vessel, broken when:
- Time gap exceeds 4 hours (configurable)
- The vessel is a detected MMSI collision (split into separate tracks)

### Track ID Format
- Normal: `{MMSI}_{segment_number}` (e.g., `219007898_0`, `219007898_1`)
- Collision: `{MMSI}_{cluster}_{segment}` (e.g., `2579999_A_0`, `2579999_B_0`)

### Track Statistics (from current data)
- Total tracks: ~50,502
- Unique vessels: ~9,809
- Average positions per track: ~11,500
- Min positions: 2 (configurable threshold)
- Date range: Dec 1, 2024 - Feb 7, 2025 (69 days)

---

## 4. Data Quality Notes

### Cleaning Applied
1. **Coordinate validation**: Only positions in Danish EEZ (54-58.5°N, 7-16°E)
2. **Outlier removal**: GPS glitches removed using velocity-skip method
3. **MMSI collision handling**: Tracks split when two vessels share same MMSI
4. **Duplicate removal**: No duplicate timestamps per vessel

### Known Considerations
- `sog` can be 0 for stationary vessels
- `heading` is often null (not all vessels transmit)
- `ship_type` is a string code (e.g., "70" for cargo)
- `dt_seconds` is 0 for the first position in each track
- Some tracks may span multiple days (track_id is consistent across files)

### MMSI Collision Tracks
32 vessels were detected with MMSI collisions (two vessels sharing one ID):
- These are split into separate tracks with `cluster_assignment` = "A" or "B"
- Filter these out if you want only clean single-vessel tracks:
```python
df = df.filter(pl.col("cluster_assignment").is_null())
```

---

## 5. Loading Data for ML

### Load All Data
```python
import polars as pl
from pathlib import Path

# Via FSx mount
data_path = Path("/mnt/fsx/cleaned")
parquet_files = list(data_path.glob("year=*/month=*/day=*/tracks.parquet"))
df = pl.concat([pl.read_parquet(f) for f in parquet_files])

# Or via S3 directly (use ** to get all years)
df = pl.read_parquet("s3://your-ais-bucket/cleaned/year=*/**/*.parquet")
```

### Load Track Catalog for Sampling
```python
catalog = pl.read_parquet("/mnt/fsx/cleaned/track_catalog.parquet")

# Filter tracks with at least 100 positions
valid_tracks = catalog.filter(pl.col("num_positions") >= 100)
print(f"Tracks with 100+ positions: {valid_tracks.height}")
```

### Load Specific Tracks
```python
# Get track IDs you want
track_ids = ["219007898_0", "220279000_0", "211833390_5"]

# Load all data and filter
df = pl.read_parquet("/mnt/fsx/cleaned/year=*/**/*.parquet")
tracks = df.filter(pl.col("track_id").is_in(track_ids))
```

### Efficient PyTorch DataLoader Pattern
```python
import torch
from torch.utils.data import Dataset, DataLoader

class AISTrackDataset(Dataset):
    def __init__(self, catalog_path, data_path, min_positions=100, max_seq_len=512):
        # Load catalog and filter
        catalog = pl.read_parquet(catalog_path)
        self.track_ids = catalog.filter(
            pl.col("num_positions") >= min_positions
        ).select("track_id").to_series().to_list()

        # Load all data (FSx makes this fast)
        self.data = pl.read_parquet(f"{data_path}/year=*/**/*.parquet")
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        track = self.data.filter(pl.col("track_id") == track_id)

        # Sort by timestamp
        track = track.sort("timestamp")

        # Extract features
        features = track.select([
            "lat", "lon", "sog", "cog", "dt_seconds"
        ]).to_numpy()

        # Truncate or pad to max_seq_len
        if len(features) > self.max_seq_len:
            features = features[:self.max_seq_len]
        elif len(features) < self.max_seq_len:
            padding = np.zeros((self.max_seq_len - len(features), features.shape[1]))
            features = np.vstack([features, padding])

        return torch.tensor(features, dtype=torch.float32)

# Usage
dataset = AISTrackDataset(
    catalog_path="/mnt/fsx/cleaned/track_catalog.parquet",
    data_path="/mnt/fsx/cleaned",
    min_positions=100
)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

---

## 6. Feature Engineering Suggestions

### Temporal Features
```python
df = df.with_columns([
    pl.col("timestamp").dt.hour().alias("hour"),
    pl.col("timestamp").dt.weekday().alias("weekday"),
    (pl.col("dt_seconds") / 60).alias("dt_minutes"),
])
```

### Velocity Components
```python
import numpy as np

df = df.with_columns([
    # Convert COG to radians and compute velocity components
    (pl.col("sog") * np.cos(np.radians(pl.col("cog")))).alias("vx"),
    (pl.col("sog") * np.sin(np.radians(pl.col("cog")))).alias("vy"),
])
```

### COG Encoding (for cyclical nature)
```python
df = df.with_columns([
    np.sin(np.radians(pl.col("cog"))).alias("cog_sin"),
    np.cos(np.radians(pl.col("cog"))).alias("cog_cos"),
])
```

### Displacement from Previous Position
```python
df = df.with_columns([
    (pl.col("lat") - pl.col("lat").shift(1).over("track_id")).alias("delta_lat"),
    (pl.col("lon") - pl.col("lon").shift(1).over("track_id")).alias("delta_lon"),
])
```

---

## 7. Common ML Tasks

### Trajectory Prediction
- **Input**: Sequence of (lat, lon, sog, cog, dt_seconds)
- **Output**: Next N positions
- **Architecture**: Transformer, LSTM, or GRU

### Anomaly Detection
- **Input**: Track sequences
- **Output**: Anomaly score per position
- **Use case**: Detecting unusual vessel behavior

### Vessel Classification
- **Input**: Track statistics or sequence embeddings
- **Output**: Vessel type prediction
- **Features**: Speed patterns, trajectory shapes, port visits

### Destination Prediction
- **Input**: Partial trajectory
- **Output**: Final destination coordinates
- **Architecture**: Encoder-decoder with attention

---

## 8. Data Volume Summary

| Metric | Value |
|--------|-------|
| Total records | 579 million |
| Total tracks | 50,502 |
| Unique vessels | 9,809 |
| Date range | Dec 1, 2024 - Feb 7, 2025 (69 days) |
| File sizes | 100-150 MB per day |
| Total size | 8.2 GB (compressed Parquet) |
| Compression | zstd |
| Raw input size | 1.11 billion records |
| Records filtered | 527.6M (outside Danish EEZ) |
| Outliers removed | 15,285 |

---

## 9. Quick Reference

### File Paths
```
S3 Bucket: your-ais-bucket
Track data: cleaned/year=YYYY/month=MM/day=DD/tracks.parquet
Catalog: cleaned/track_catalog.parquet
```

### Key Columns for ML
```
Primary: track_id, timestamp, lat, lon, sog, cog, dt_seconds
Optional: heading, ship_type
Metadata: mmsi, name, imo, callsign
Filter: cluster_assignment (null = clean track)
```

### Polars Quick Commands
```python
# Read all data
df = pl.read_parquet("path/**/*.parquet")

# Get unique tracks
tracks = df.select("track_id").unique()

# Group by track
for track_df in df.partition_by("track_id"):
    process(track_df)

# Filter by track length
long_tracks = df.group_by("track_id").agg(pl.count()).filter(pl.col("count") >= 100)
```
