import pandas as pd
import os
import sys
from pathlib import Path

def find_long_tracks(data_dir: str, num_tracks: int) -> pd.DataFrame:
    """
    Finds and prints the longest tracks based on geographical span.
    """
    print(f"Analyzing tracks in {data_dir}...")
    all_files = [os.path.join(root, f) for root, _, files in os.walk(data_dir) for f in files if f.endswith('.parquet')]
    
    if not all_files:
        print("No parquet files found in the directory.")
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

def main():
    # Go up three levels from the script directory to find the project root
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    data_directory = os.path.join(project_root, 'data', '03_primary', 'resampled_ais_data')

    if not os.path.exists(data_directory):
        print(f"Error: Data directory not found at '{data_directory}'")
        print("Please ensure the pipeline has been run and the directory structure is correct.")
        sys.exit(1)
        
    print(f"Analyzing tracks in {data_directory}...")
    top_tracks = find_long_tracks(data_directory, 4)
    
    if top_tracks.empty:
        print("\nCould not find any tracks to analyze.")
        return
        
    print("\n--- Top 4 Longest Tracks by Geographical Span ---")
    
    # Print the top 4 MMSIs and their track IDs
    for index, row in top_tracks.iterrows():
        print(f"MMSI: {row['mmsi']}, Track ID: {row['track_id']}")

if __name__ == '__main__':
    main() 