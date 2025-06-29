import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import argparse
import logging
import polars as pl

def load_mmsi_data(mmsi: int, data_dir: str) -> pd.DataFrame:
    """
    Loads all data for a given MMSI from a partitioned parquet directory
    using Polars for efficient filtering.
    """
    try:
        # Lazily scan the dataset and filter for the specific MMSI *before* loading data.
        # This is massively more efficient than loading all data.
        lazy_df = pl.scan_parquet(os.path.join(data_dir, "**/*.parquet"))
        
        mmsi_df = (
            lazy_df.filter(pl.col('mmsi') == mmsi)
            .sort('timestamp')
            .collect()
        )
        
        if mmsi_df.is_empty():
            return pd.DataFrame()
            
        return mmsi_df.to_pandas()
        
    except pl.exceptions.NoDataError:
        logging.warning("No data files found in %s", data_dir)
        return pd.DataFrame()
    except Exception as e:
        logging.error("Failed to load data for MMSI %d from %s. Error: %s", mmsi, data_dir, e)
        return pd.DataFrame()

def plot_track_comparison(ax, original_df, resampled_df, mmsi):
    """
    Plots the original and resampled tracks on a given matplotlib axis.
    """
    # Plot each track_id group separately to avoid connecting disjointed tracks
    for i, (track_id, group) in enumerate(original_df.groupby('track_id')):
        # Ensure data is sorted by time before plotting
        group = group.sort_values('timestamp')
        label = 'Original' if i == 0 else None # Only add label once
        ax.plot(group['lon'], group['lat'], 'o-', label=label, markersize=3, color='blue')
        # Add start and end points for each track
        ax.plot(group['lon'].iloc[0], group['lat'].iloc[0], 'go', markersize=7, label='Track Start' if i == 0 else None)
        ax.plot(group['lon'].iloc[-1], group['lat'].iloc[-1], 'yo', markersize=7, label='Track End' if i == 0 else None)

    for i, (track_id, group) in enumerate(resampled_df.groupby('track_id')):
        # Ensure data is sorted by time before plotting
        group = group.sort_values('timestamp')
        label = 'Resampled (10 min)' if i == 0 else None # Only add label once
        ax.plot(group['lon'], group['lat'], '.-', label=label, markersize=5, color='red', alpha=0.7)

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