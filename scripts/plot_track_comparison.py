import pandas as pd
import os
import matplotlib.pyplot as plt
import sys

def load_mmsi_data(mmsi, data_dir):
    """
    Loads all data for a given MMSI from a partitioned parquet directory.
    """
    all_files = [os.path.join(root, f) for root, _, files in os.walk(data_dir) for f in files if f.endswith('.parquet')]
    
    mmsi_df_parts = []
    for file_path in all_files:
        try:
            df = pd.read_parquet(file_path, columns=['mmsi', 'timestamp', 'lat', 'lon'])
            mmsi_df_parts.append(df[df['mmsi'] == mmsi])
        except Exception as e:
            print(f"Warning: Could not read {file_path}. Error: {e}", file=sys.stderr)

    if not mmsi_df_parts:
        return pd.DataFrame()

    return pd.concat(mmsi_df_parts).sort_values('timestamp').reset_index(drop=True)

def plot_track_comparison(ax, original_df, resampled_df, mmsi):
    """
    Plots the original and resampled tracks on a given matplotlib axis.
    """
    ax.plot(original_df['lon'], original_df['lat'], 'o-', label='Original', markersize=3, color='blue')
    ax.plot(resampled_df['lon'], resampled_df['lat'], '.-', label='Resampled (10 min)', markersize=5, color='red', alpha=0.7)
    
    # Add start and end points for clarity
    if not original_df.empty:
        ax.plot(original_df['lon'].iloc[0], original_df['lat'].iloc[0], 'go', markersize=10, label='Start')
        ax.plot(original_df['lon'].iloc[-1], original_df['lat'].iloc[-1], 'yo', markersize=10, label='End')

    ax.set_title(f'Vessel Track Comparison for MMSI: {mmsi}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True)
    
    # Enforce equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

def main():
    """
    Main function to generate and save the comparison plot.
    """
    # These are the 4 longest tracks identified by find_long_tracks.py
    mmsi_list = [265022000, 636018490, 366698000, 354643000]
    
    # Setup paths
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    original_data_dir = os.path.join(project_root, 'data', '02_intermediate', 'cleaned_partitioned_ais')
    resampled_data_dir = os.path.join(project_root, 'data', '03_primary', 'resampled_ais_data')
    output_dir = os.path.join(project_root, 'data', '04_reporting')
    os.makedirs(output_dir, exist_ok=True)

    for mmsi in mmsi_list:
        print(f"Processing MMSI: {mmsi}")
        try:
            # Load the data for the specific MMSI
            original_df = load_mmsi_data(mmsi, original_data_dir)
            resampled_df = load_mmsi_data(mmsi, resampled_data_dir)

            if original_df.empty or resampled_df.empty:
                print(f"No data found for MMSI {mmsi}. Skipping.")
                continue

            # Generate and save the plot
            fig, ax = plt.subplots(figsize=(14, 10))
            plot_track_comparison(ax, original_df, resampled_df, mmsi)
            
            output_path = os.path.join(output_dir, f'ais_track_before_after_interp_{mmsi}.png')
            plt.savefig(output_path, bbox_inches='tight')
            plt.close(fig) # Close the figure to free memory
            print(f"Saved plot to {output_path}")

        except Exception as e:
            print(f"Could not process or plot MMSI {mmsi}. Error: {e}")


if __name__ == "__main__":
    main() 