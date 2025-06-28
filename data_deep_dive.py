import pandas as pd
import os
from geopy.distance import geodesic

def load_track_from_file(file_path, mmsi):
    """Loads and prepares data for a specific MMSI from a single parquet file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(file_path)
        df['mmsi'] = df['mmsi'].astype(float)
        mmsi_df = df[df['mmsi'] == float(mmsi)].copy()
        if not mmsi_df.empty:
            mmsi_df['timestamp'] = pd.to_datetime(mmsi_df['timestamp'])
            mmsi_df = mmsi_df.sort_values(by='timestamp').reset_index(drop=True)
        return mmsi_df
    except Exception as e:
        print(f"Could not process file {file_path}: {e}")
        return pd.DataFrame()

def main():
    mmsi_to_plot = 2182807.0
    original_file = 'data/cleaned_partitioned_ais/year=2025/month=2/day=1/part-0.parquet'
    resampled_file = 'data/resampled_ais_data/year=2025/month=2/day/1/part-0.parquet'

    # Correct file path
    resampled_file = 'data/resampled_ais_data/year=2025/month=2/day=1/part-0.parquet'

    print("Loading data...")
    original_df = load_track_from_file(original_file, mmsi_to_plot)
    resampled_df = load_track_from_file(resampled_file, mmsi_to_plot)

    if original_df.empty or resampled_df.empty:
        print("Could not find data for the specified MMSI.")
        return

    # Sort by timestamp for merging
    original_df.sort_values('timestamp', inplace=True)
    resampled_df.sort_values('timestamp', inplace=True)

    print("Aligning resampled points to nearest original points...")
    # Use merge_asof to find the nearest original point for each resampled point
    aligned_df = pd.merge_asof(
        resampled_df,
        original_df,
        on='timestamp',
        by='mmsi',
        direction='nearest',
        suffixes=('_resampled', '_original')
    )

    # Calculate the distance between the resampled point and its nearest original neighbor
    distances = []
    for _, row in aligned_df.iterrows():
        pt1 = (row['lat_resampled'], row['lon_resampled'])
        pt2 = (row['lat_original'], row['lon_original'])
        distances.append(geodesic(pt1, pt2).kilometers)
    
    aligned_df['distance_km'] = distances

    print("\\n--- Data Alignment and Distance Verification ---")
    print("This table shows each resampled point, its nearest neighbor in the original data, and the distance between them.")
    print("If the resampling was correct, the distances should be very small.\\n")
    
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.width', 200)

    print(aligned_df[[
        'timestamp', 'lat_resampled', 'lon_resampled',
        'lat_original', 'lon_original', 'distance_km'
    ]])

    print(f"\\nSummary of distances (km):")
    print(aligned_df['distance_km'].describe())


if __name__ == '__main__':
    main() 