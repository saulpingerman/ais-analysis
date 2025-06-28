# AIS Data Processing and Analysis Pipeline

This repository contains a collection of scripts and notebooks designed to fetch, clean, partition, and analyze maritime Automatic Identification System (AIS) data. The primary goal is to process raw vessel tracking data into a clean, analysis-ready format suitable for visualization and further modeling.

## Key Features

- **Data Ingestion**: Fetch weekly AIS data archives from source.
- **Efficient Formatting**: Convert raw CSV data to the efficient and compressed Parquet format.
- **Scalable Partitioning**: Partition the data by date (`year/month/day`) for scalable and efficient queries.
- **Data Cleaning**: Standardize and clean data to handle common errors and inconsistencies.
- **Time-Series Resampling**: Resample vessel tracks to a fixed time interval (e.g., every 10 minutes) using linear interpolation.
- **Visualization**: Generate plots to compare original and resampled vessel tracks for verification and analysis.

## Project Structure

```
ais-analysis/
│
├── scripts/
│   ├── ais_data_cleaner.py           # Cleans and splits raw partitioned data
│   ├── resample_ais_data_simple.py   # Main script for track resampling
│   ├── plot_track_comparison.py      # Script to generate comparison plots
│   └── test_cleaner_logic.py         # Test suite for the data cleaner
│
├── notebooks/
│   └── (Exploratory notebooks)
│
├── figures/
│   └── (Output plots for verification)
│
└── README.md
```

## Workflow / How to Use

The main workflow consists of two primary steps: cleaning the raw data and then resampling the cleaned tracks to a uniform time interval.

### 1. Clean Raw Data
The `ais_data_cleaner.py` script processes a directory of partitioned Parquet files. It filters out bad data points, removes duplicates, and splits long voyages into distinct tracks based on time gaps. It maintains state between runs to ensure track IDs are consistent across multiple days.

```bash
python scripts/ais_data_cleaner.py \
  --input path/to/raw_partitioned_data \
  --output path/to/cleaned_data \
  --speed_thresh 80.0 \
  --gap_hours 6.0
```

### 2. Resample Tracks
Once the data is cleaned, the `resample_ais_data_simple.py` script can be used to interpolate the tracks to a uniform 10-minute interval. This is useful for time-series analysis.

```bash
python scripts/resample_ais_data_simple.py \
  --input path/to/cleaned_data \
  --output path/to/resampled_data
```

### 3. Verify and Analyze
Use the `plot_track_comparison.py` script to generate figures comparing the original and resampled tracks for a specific vessel. This is essential for visually verifying the quality of the interpolation.

```bash
python scripts/plot_track_comparison.py \
  --mmsi 123456789 \
  --original_path path/to/cleaned_data \
  --resampled_path path/to/resampled_data \
  --output_dir figures/
``` 