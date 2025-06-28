import pandas as pd
import os
import sys
from pathlib import Path
import argparse
import logging

def find_long_tracks(data_dir: str, num_tracks: int) -> pd.DataFrame:
    """
    Finds and prints the longest tracks based on geographical span.
    """
    logging.info("Analyzing tracks in %s...", data_dir)
    all_files = [os.path.join(root, f) for root, _, files in os.walk(data_dir) for f in files if f.endswith('.parquet')]
    
    if not all_files:
        logging.warning("No parquet files found in the directory.")
        return pd.DataFrame()

    # The type checker struggles with a list of paths, so we ignore it.
    df = pd.read_parquet(all_files) # type: ignore

    # Calculate geographical span for each track
    span = df.groupby(['mmsi', 'track_id']).agg(
        lat_span=('lat', lambda x: x.max() - x.min()),
        lon_span=('lon', lambda x: x.max() - x.min())
    ).reset_index() # Move mmsi and track_id from index to column
    
    span['total_span'] = span['lat_span'] + span['lon_span']
    
    return span.sort_values(by='total_span', ascending=False).head(num_tracks)

def main(args):
    data_directory = args.input
    num_tracks = args.num_tracks

    if not os.path.exists(data_directory):
        logging.error("Data directory not found at '%s'", data_directory)
        sys.exit(1)
        
    top_tracks = find_long_tracks(data_directory, num_tracks)
    
    if top_tracks.empty:
        logging.info("Could not find any tracks to analyze.")
        return
        
    logging.info("--- Top %d Longest Tracks by Geographical Span ---", num_tracks)
    
    # Print the top 4 MMSIs and their track IDs
    for index, row in top_tracks.iterrows():
        logging.info("MMSI: %s, Track ID: %s", row['mmsi'], row['track_id'])

if __name__ == '__main__':
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
        default=4,
        help="The number of top tracks to find. (Default: 4)",
    )
    args = parser.parse_args()
    main(args) 