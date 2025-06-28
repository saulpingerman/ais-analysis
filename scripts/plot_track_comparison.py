import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import argparse
import logging

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
            logging.warning("Could not read or process %s. Error: %s", file_path, e)

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

def main(args):
    """
    Main function to generate and save the comparison plot for a single MMSI.
    """
    mmsi = args.mmsi
    original_data_dir = args.original_path
    resampled_data_dir = args.resampled_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Processing MMSI: %d", mmsi)
    try:
        # Load the data for the specific MMSI
        original_df = load_mmsi_data(mmsi, original_data_dir)
        resampled_df = load_mmsi_data(mmsi, resampled_data_dir)

        if original_df.empty:
            logging.warning("No original data found for MMSI %d. Skipping.", mmsi)
            return
        if resampled_df.empty:
            logging.warning("No resampled data found for MMSI %d. Skipping.", mmsi)
            return

        # Generate and save the plot
        fig, ax = plt.subplots(figsize=(14, 10))
        plot_track_comparison(ax, original_df, resampled_df, mmsi)
        
        output_path = os.path.join(output_dir, f'ais_track_before_after_interp_{mmsi}.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        logging.info("Saved plot to %s", output_path)

    except Exception as e:
        logging.error("Could not process or plot MMSI %d. Error: %s", mmsi, e)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Generate a plot comparing original and resampled vessel tracks."
    )
    parser.add_argument(
        "--mmsi",
        type=int,
        required=True,
        help="The MMSI of the vessel to plot.",
    )
    parser.add_argument(
        "--original_path",
        type=str,
        required=True,
        help="Path to the root directory of the original (cleaned) partitioned data.",
    )
    parser.add_argument(
        "--resampled_path",
        type=str,
        required=True,
        help="Path to the root directory of the resampled partitioned data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output plot.",
    )
    args = parser.parse_args()
    main(args) 