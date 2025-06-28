import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_track_from_file(file_path, mmsi):
    """
    Loads data for a specific MMSI from a single parquet file.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(file_path)
        mmsi_df = df[df['mmsi'] == mmsi].copy()
        if not mmsi_df.empty:
            mmsi_df['timestamp'] = pd.to_datetime(mmsi_df['timestamp'])
            mmsi_df = mmsi_df.sort_values(by='timestamp')
        return mmsi_df
    except Exception as e:
        print(f"Could not process file {file_path}: {e}")
        return pd.DataFrame()

def plot_ais_track_comparison(original_df, resampled_df, mmsi_to_plot):
    # --- Filter data to a smaller time window for clarity ---
    start_time = '2025-02-01 04:00:00'
    end_time = '2025-02-01 07:00:00'
    
    original_df_filtered = original_df[(original_df['timestamp'] >= start_time) & (original_df['timestamp'] <= end_time)]
    resampled_df_filtered = resampled_df[(resampled_df['timestamp'] >= start_time) & (resampled_df['timestamp'] <= end_time)]
    # --- End filter ---

    # --- START DIAGNOSTIC PRINT ---
    print("--- Original Data to Plot ---")
    print(original_df_filtered[['lat', 'lon']].head())
    print("\n--- Resampled Data to Plot ---")
    print(resampled_df_filtered[['lat', 'lon']].head())
    print("---------------------------------")
    # --- END DIAGNOSTIC PRINT ---

    # Reset index AFTER filtering
    original_df_filtered = original_df_filtered.reset_index(drop=True)
    resampled_df_filtered = resampled_df_filtered.reset_index(drop=True)

    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Plot original track
    plt.plot(original_df_filtered['lon'], original_df_filtered['lat'], 'bo-', label='Original Track', markersize=4, alpha=0.7, linewidth=1)
    
    # Plot resampled track
    plt.plot(resampled_df_filtered['lon'], resampled_df_filtered['lat'], 'rs--', label='Resampled Track (10 min)', markersize=6, linewidth=2)
    
    plt.title(f'AIS Track Comparison for MMSI: {int(mmsi_to_plot)} (Zoomed View)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_filename = 'ais_track_comparison_zoomed.png'
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

def main():
    mmsi_to_check = 230700000.0
    original_file = 'data/cleaned_partitioned_ais/year=2025/month=2/day=1/part-0.parquet'
    resampled_file = 'data/resampled_ais_data/year=2025/month=2/day=1/part-0.parquet'
    
    print(f"Loading original data for MMSI: {mmsi_to_check}...")
    original_df = load_track_from_file(original_file, mmsi_to_check)
    
    print(f"Loading resampled data for MMSI: {mmsi_to_check}...")
    resampled_df = load_track_from_file(resampled_file, mmsi_to_check)
    
    if not original_df.empty and not resampled_df.empty:
        plot_ais_track_comparison(original_df, resampled_df, mmsi_to_check)
    else:
        print("Could not generate plot due to missing data.")

if __name__ == '__main__':
    main() 