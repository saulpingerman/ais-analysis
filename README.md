# AIS Data Processing Pipeline

A high-performance, production-ready pipeline for processing AIS (Automatic Identification System) maritime vessel tracking data from the Danish Maritime Authority.

## Features

- **High-throughput processing**: Handles 100M+ records efficiently using Polars
- **MMSI collision detection**: Identifies and separates tracks when multiple vessels share the same MMSI
- **Velocity-based outlier removal**: Detects and removes GPS glitches using the velocity-skip method
- **Cross-file track continuity**: Maintains consistent track IDs across daily data files
- **Partitioned Parquet output**: Optimized for downstream ML training with FSx for Lustre
- **Resumable processing**: Checkpoint-based recovery for long-running jobs
- **S3-native**: Direct integration with AWS S3 for input/output

## Installation

```bash
# Clone the repository
git clone https://github.com/saulpingerman/ais-analysis.git
cd ais-analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

## Quick Start

### 1. Configure S3 bucket

Edit `config/production.yaml`:

```yaml
storage:
  s3_bucket: "your-ais-bucket"
  raw_prefix: "raw/"
  cleaned_prefix: "cleaned/"
  state_prefix: "state/"
```

### 2. Download AIS data

```bash
# Download one month of data from Danish Maritime Authority
python scripts/download_ais_data.py --start-date 2024-01-01 --end-date 2024-01-31
```

### 3. Run the processing pipeline

```bash
# Process all files
python run_pipeline.py

# Process with file limit (for testing)
python run_pipeline.py --max-files 2

# Resume from checkpoint
python run_pipeline.py --resume
```

## Project Structure

```
ais-analysis/
├── config/
│   └── production.yaml       # Production configuration
├── src/
│   └── ais_pipeline/
│       ├── cleaning/         # Data cleaning modules
│       │   ├── validator.py  # Basic validation (bounds, nulls)
│       │   ├── outliers.py   # Velocity-skip outlier detection
│       │   ├── collision.py  # MMSI collision detection (DBSCAN)
│       │   └── segmentation.py
│       ├── state/            # State management
│       │   ├── continuity.py # Cross-file track continuity
│       │   └── checkpoint.py # Processing checkpoints
│       ├── io/               # I/O modules
│       │   ├── reader.py     # ZIP/CSV reading from S3
│       │   └── writer.py     # Partitioned Parquet output
│       ├── utils/            # Utilities
│       │   ├── geo.py        # Haversine distance
│       │   └── velocity.py   # Speed calculations
│       ├── config.py         # Configuration management
│       ├── pipeline.py       # Main orchestration
│       └── cli.py            # Command-line interface
├── scripts/                  # Utility scripts
│   ├── download_ais_data.py  # DMA data downloader
│   └── s3_ais_processor.py   # Legacy processor
├── tests/                    # Test suite
├── run_pipeline.py           # Main entry point
└── pyproject.toml
```

## Data Cleaning Pipeline

The pipeline applies the following cleaning steps in order:

### 1. Basic Validation
- Remove null coordinates
- Filter to Danish EEZ bounding box (54-58.5N, 7-16E)
- Remove invalid SOG values (>102.3 knots)
- Remove duplicate timestamps per MMSI

### 2. MMSI Collision Detection
Detects when two vessels share the same MMSI identifier by:
- Identifying impossible "teleportation" patterns
- Clustering positions using DBSCAN
- Detecting bounce patterns between clusters
- Splitting into separate tracks (MMSI_A, MMSI_B)

### 3. Single Outlier Removal
Uses the velocity-skip method:
- For each position, calculates velocity to/from neighbors
- If both incoming and outgoing velocities exceed threshold but skip velocity is reasonable, the position is an outlier

### 4. Track Segmentation
- Breaks tracks where time gap exceeds threshold (default: 4 hours)
- Assigns unique track IDs: `{MMSI}_{segment_number}`
- Maintains continuity across file boundaries

## Output Format

### Parquet Files
```
s3://bucket/cleaned/
├── year=2024/month=01/day=01/tracks.parquet
├── year=2024/month=01/day=02/tracks.parquet
└── track_catalog.parquet
```

### Schema
| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Position timestamp (UTC) |
| mmsi | int64 | Vessel identifier |
| track_id | string | Unique track ID |
| lat | float64 | Latitude |
| lon | float64 | Longitude |
| sog | float64 | Speed over ground (knots) |
| cog | float64 | Course over ground (degrees) |
| dt_seconds | float32 | Time since previous position |
| cluster_assignment | string | A/B for collision tracks |

### Track Catalog
Provides metadata for efficient ML training:
- Track start/end times
- Number of positions
- Duration in hours

## Configuration

Key parameters in `config/production.yaml`:

```yaml
cleaning:
  track_gap_hours: 4.0          # Gap to start new track
  max_velocity_knots: 50.0      # Max speed for outlier detection
  collision:
    distance_threshold_km: 50.0  # Min distance for collision check
    min_bounce_count: 3          # Min bounces to confirm collision

output:
  compression: "zstd"
  row_group_size: 100000
```

## Performance

Tested on r6i.4xlarge (16 vCPU, 128 GB RAM):
- **Throughput**: ~156,000 records/second
- **69 days of data** (1.1B records): ~2 hours

### Latest Processing Run Statistics
| Metric | Value |
|--------|-------|
| Files processed | 69 days (Dec 1, 2024 - Feb 7, 2025) |
| Input records | 1.11 billion |
| Output records | 579 million |
| Records filtered (outside EEZ) | 527.6 million |
| Outliers removed | 15,285 |
| MMSI collisions detected | 32 |
| Unique vessels | 9,809 |
| Unique tracks | 50,502 |
| Output size | 8.2 GB (compressed Parquet) |

## Legacy Scripts

The `scripts/` directory contains the original processing scripts:

- **`s3_ais_processor.py`** - Original S3-based processor
- **`download_ais_data.py`** - Download data from Danish Maritime Authority
- **`run_s3_pipeline.py`** - Interactive examples

## AWS Requirements

- **S3 bucket** with appropriate IAM permissions
- **EC2 instance** with IAM role for S3 access
- Recommended: r6i.4xlarge or similar memory-optimized instance

### IAM Permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*"
      ]
    }
  ]
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Data Source

AIS data provided by the [Danish Maritime Authority](http://aisdata.ais.dk/).
