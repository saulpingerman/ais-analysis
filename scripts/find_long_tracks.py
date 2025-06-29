import polars as pl
import os
import sys
from pathlib import Path
import argparse
import logging
import yaml

def load_config(path='config.yaml'):
    """Loads the YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def find_long_tracks(data_dir: str, num_tracks: int) -> pl.DataFrame:
    """
    Finds and prints the longest tracks based on geographical span using Polars for memory efficiency.
    """
    logging.info("Analyzing tracks in %s...", data_dir)
    
    try:
        # Lazily scan the dataset to avoid loading all data into memory
        lazy_df = pl.scan_parquet(os.path.join(data_dir, "**/*.parquet"))
    except pl.exceptions.NoDataError:
        logging.warning("No parquet files found in the directory.")
        return pl.DataFrame()

    # Calculate geographical span for each track using lazy expressions
    span_df = (
        lazy_df.group_by('track_id')
        .agg(
            pl.first('mmsi').alias('mmsi'),
            (pl.max('lat') - pl.min('lat')).alias('lat_span'),
            (pl.max('lon') - pl.min('lon')).alias('lon_span')
        )
        .with_columns(
            (pl.col('lat_span') + pl.col('lon_span')).alias('total_span')
        )
        .sort(by='total_span', descending=True)
        .head(num_tracks)
    )
    
    # Collect the final small result into memory
    return span_df.collect()

def main(args):
    data_directory = args.input
    num_tracks = args.num_tracks

    if not os.path.exists(data_directory):
        logging.error("Data directory not found at '%s'", data_directory)
        sys.exit(1)
        
    top_tracks = find_long_tracks(data_directory, num_tracks)
    
    if top_tracks.is_empty():
        logging.info("Could not find any tracks to analyze.")
        return
        
    logging.info("--- Top %d Longest Tracks by Geographical Span ---", num_tracks)
    
    # Print the top tracks
    for row in top_tracks.iter_rows(named=True):
        logging.info("MMSI: %s, Track ID: %s", row['mmsi'], row['track_id'])

if __name__ == '__main__':
    # Load configuration from YAML
    config = load_config()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Find the longest vessel tracks in a dataset based on geographical span."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Root directory of the partitioned data to analyze.",
    )
    parser.add_argument(
        "--num_tracks",
        "-n",
        type=int,
        default=config.get('num_tracks', 4),
        help=f"The number of top tracks to find. (Default: {config.get('num_tracks', 4)})",
    )
    args = parser.parse_args()
    main(args) 