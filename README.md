# AIS Data Processing Pipeline

A high-performance pipeline for processing AIS (Automatic Identification System) maritime vessel tracking data from the Danish Maritime Authority. Reads zip files from a local directory and produces partitioned Parquet, ready for downstream ML training.

## Features

- **High-throughput processing**: Handles 100M+ records efficiently using Polars
- **MMSI collision detection**: Identifies and separates tracks when multiple vessels share the same MMSI
- **Velocity-based outlier removal**: Detects and removes GPS glitches using the velocity-skip method
- **Cross-file track continuity**: Maintains consistent track IDs across daily data files
- **Partitioned Parquet output**: `year=YYYY/month=MM/day=DD/tracks.parquet`
- **Resumable processing**: Local checkpoint-based recovery for long-running jobs

## Installation

```bash
git clone https://github.com/saulpingerman/ais-analysis.git
cd ais-analysis

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

## Quick Start

### 1. Configure paths (optional)

Default storage layout is `~/data/ais/{raw,clean,state}`. Override in `config/production.yaml` if needed:

```yaml
storage:
  raw_dir:   "~/data/ais/raw"
  clean_dir: "~/data/ais/clean"
  state_dir: "~/data/ais/state"
```

### 2. Download AIS data

```bash
# One day (daily files exist from March 2024 onwards)
python scripts/download_ais_data.py --start-date 2024-01-15 --end-date 2024-01-15

# Full year (mix of monthly + daily handled automatically)
python scripts/download_ais_data.py --start-date 2024-01-01 --end-date 2024-12-31
```

Files land at `<raw_dir>/<year>/aisdk-<date>.zip`.

### 3. Run the processing pipeline

```bash
# Process all files under raw_dir
python run_pipeline.py

# Process the first N files (smoke test)
python run_pipeline.py --max-files 1

# Resume from checkpoint
python run_pipeline.py --resume
```

## Project Structure

```
ais-analysis/
├── config/
│   └── production.yaml       # Pipeline configuration
├── src/
│   └── ais_pipeline/
│       ├── cleaning/         # Cleaning modules
│       │   ├── validator.py  # Basic validation (bounds, nulls)
│       │   ├── outliers.py   # Velocity-skip outlier detection
│       │   ├── collision.py  # MMSI collision detection (DBSCAN)
│       │   └── segmentation.py
│       ├── state/            # State (local JSON)
│       │   ├── continuity.py # Cross-file track continuity
│       │   └── checkpoint.py # Processing checkpoints
│       ├── io/               # I/O modules (local FS)
│       │   ├── reader.py     # ZIP/CSV reading
│       │   └── writer.py     # Partitioned Parquet output
│       ├── utils/            # Haversine, velocity helpers
│       ├── config.py         # Configuration loading
│       ├── pipeline.py       # Orchestration
│       └── cli.py            # `ais-pipeline` CLI
├── scripts/
│   └── download_ais_data.py  # DMA downloader
├── tests/
├── run_pipeline.py           # Main entry point
└── pyproject.toml
```

## Data Cleaning Pipeline

1. **Basic Validation** — null filtering, Danish EEZ bounding box (54–58.5N, 7–16E), invalid SOG removal, duplicate timestamp dedup per MMSI.
2. **MMSI Collision Detection** — detects vessels sharing an MMSI by clustering positions (DBSCAN) and identifying bounce patterns; splits into separate tracks.
3. **Single Outlier Removal** — velocity-skip method: drop a position if the in/out velocities exceed threshold but the skip velocity is plausible.
4. **Track Segmentation** — splits when time gap exceeds threshold (default 4h); track IDs `{MMSI}_{segment}`; continuity preserved across files.
5. **Track Validation** — drop tracks shorter than `min_track_points`.

## Output Format

```
<clean_dir>/
├── year=2024/month=01/day=15/tracks.parquet
├── year=2024/month=01/day=16/tracks.parquet
└── track_catalog.parquet
```

### Schema
| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Position timestamp (UTC) |
| mmsi | int64 | Vessel identifier |
| track_id | string | Unique track ID (`{mmsi}_{segment}`) |
| lat | float64 | Latitude |
| lon | float64 | Longitude |
| sog | float64 | Speed over ground (knots) |
| cog | float64 | Course over ground (degrees) |
| dt_seconds | float32 | Time since previous position |
| cluster_assignment | string | A/B for MMSI-collision tracks |

## Configuration

Key parameters in `config/production.yaml`:

```yaml
cleaning:
  track_gap_hours: 4.0
  max_velocity_knots: 50.0
  collision:
    distance_threshold_km: 50.0
    min_bounce_count: 3

output:
  compression: "zstd"
  row_group_size: 100000
```

## License

MIT — see [LICENSE](LICENSE).

## Data Source

AIS data provided by the [Danish Maritime Authority](http://aisdata.ais.dk/).
