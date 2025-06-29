# AIS Data Analysis Pipeline

This project provides a complete, high-performance pipeline for ingesting, cleaning, processing, and analyzing raw Automatic Identification System (AIS) data. The workflow transforms large volumes of raw CSV data from ZIP archives into a clean, partitioned, and analysis-ready dataset in Parquet format.

The entire pipeline is built with modern, memory-efficient tooling, primarily using the **Polars** DataFrame library to handle datasets larger than available RAM.

## Key Technologies

- **Python**: The core language for all scripting.
- **Polars**: Used for high-performance, memory-efficient DataFrame manipulation, enabling the processing of large datasets on commodity hardware.
- **Pandas / Matplotlib**: Used for final data resampling and plotting.
- **PyArrow**: The underlying engine for efficient Parquet file handling.

## Project Structure

Our data processing pipeline follows a clear, staged directory structure:

```
data/
├── 01_raw/             # Raw, unprocessed source data (e.g., ZIP archives)
├── 02_intermediate/    # Intermediate data formats (e.g., raw partitioned Parquet)
├── 03_primary/         # Cleaned, analysis-ready datasets
└── 04_reporting/       # Final outputs like plots and reports
```

The scripts that operate on this data are organized as follows:

```
repos/ais-analysis/
│
├── scripts/
│   ├── zip_to_parquet.py             # Ingests ZIPs -> Raw Parquet
│   ├── ais_data_cleaner.py           # Cleans raw data and creates tracks
│   ├── resample_ais_data_simple.py   # Resamples tracks to 10-min intervals
│   ├── find_long_tracks.py           # Finds longest tracks by distance
│   └── plot_track_comparison.py      # Generates before/after comparison plots
│
└── README.md
```

## End-to-End Workflow

Follow these steps in order to run the full data pipeline.

### Step 1: Ingest Raw Data

Place your raw AIS `.zip` archives into the `data/01_raw/ais_dk/` directory. Then, run the `zip_to_parquet.py` script to convert the CSVs inside the ZIPs into a partitioned Parquet dataset. This script is memory-safe and can handle very large archives.

```bash
python3 repos/ais-analysis/scripts/zip_to_parquet.py \
  --source-dir data/01_raw/ais_dk/ \
  --dest-dir data/02_intermediate/raw_partitioned_ais/
```

### Step 2: Clean and Create Tracks

Run the `ais_data_cleaner.py` script to process the raw Parquet data. This step performs several key actions:
- Removes duplicate and invalid position reports.
- Filters out data points that imply impossible speeds (>80 knots).
- Intelligently groups position reports into unique voyages (`track_id`), correctly handling tracks that span across multiple days.

```bash
python3 repos/ais-analysis/scripts/ais_data_cleaner.py \
  --input data/02_intermediate/raw_partitioned_ais/ \
  --output data/03_primary/cleaned_partitioned_ais/ \
  --gap_hours 6.0
```

### Step 3: Resample Tracks to 10-Minute Intervals

Run the `resample_ais_data_simple.py` script to regularize the time-series data. It takes the cleaned tracks and interpolates them to create a consistent 10-minute interval between each data point.

```bash
python3 repos/ais-analysis/scripts/resample_ais_data_simple.py \
  --input data/03_primary/cleaned_partitioned_ais/ \
  --output data/03_primary/10m_cleaned_partitioned_ais/
```

### Step 4: Analyze and Visualize

With the data fully processed, you can now perform analysis.

**A. Find the Longest Tracks**

Use `find_long_tracks.py` to identify the longest voyages in the dataset based on their geographical span.

```bash
python3 repos/ais-analysis/scripts/find_long_tracks.py \
  --input data/03_primary/10m_cleaned_partitioned_ais/ \
  --num_tracks 4
```

**B. Generate Comparison Plots**

Use `plot_track_comparison.py` to create a visual comparison of a specific vessel's track before and after resampling. This is crucial for verifying that the interpolation process behaved as expected.

```bash
# First, clear any old plots
rm -f data/04_reporting/*.png

# Get the top 4 MMSIs (example)
MMSI_LIST="220253000 220464000 636018490 245546000"

# Loop and generate a plot for each one
for mmsi in $MMSI_LIST; do
    echo "--- Plotting MMSI: $mmsi ---"
    python3 repos/ais-analysis/scripts/plot_track_comparison.py \
        --mmsi "$mmsi" \
        --original_path data/03_primary/cleaned_partitioned_ais/ \
        --resampled_path data/03_primary/10m_cleaned_partitioned_ais/ \
        --output_dir data/04_reporting/
done
```
The final plots will be saved in the `data/04_reporting/` directory. 