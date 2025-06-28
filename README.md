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
│   ├── fetch_week.sh           # Fetches raw data archives
│   ├── csv_to_parquet.py       # Converts CSV to Parquet
│   ├── ais_partitioning.py     # Partitions data by date
│   └── ais_data_cleaner.py     # Cleans partitioned data
│
├── notebooks/
│   └── basic_viz_check.ipynb   # Notebook for exploratory analysis and visualization
│
├── figures/
│   └── ais_track_comparison.png # Example output plots
│
├── resample_ais_data_simple.py # Main script for track resampling
├── plot_comparison.py        # Script to generate comparison plots
└── ...                       # Other verification and analysis scripts
```

## Workflow / How to Use

The scripts are designed to be run as a pipeline to process AIS data from its raw form to a cleaned, resampled dataset.

### 1. Fetch Data
Use the `fetch_week.sh` script to download the raw data archives.
```bash
./scripts/fetch_week.sh
```

### 2. Convert and Partition
Convert the raw CSV files to Parquet and then partition them for efficient access.
```bash
python scripts/csv_to_parquet.py
python scripts/ais_partitioning.py
```

### 3. Clean Data
Run the cleaning script on the partitioned data.
```bash
python scripts/ais_data_cleaner.py
```

### 4. Resample Tracks
Perform the time-based resampling on the cleaned data.
```bash
python resample_ais_data_simple.py
```

### 5. Analyze and Visualize
Use the `plot_comparison.py` script or the notebooks in the `/notebooks` directory to inspect the results and generate figures.
```bash
python plot_comparison.py
```
This will produce comparison plots, which can be found in the `/figures` directory. 