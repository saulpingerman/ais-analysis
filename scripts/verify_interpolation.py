import polars as pl
import argparse
import logging
from pathlib import Path

def analyze_mmsi_interpolation(mmsi: int, original_dir: Path, resampled_dir: Path):
    """
    Analyzes the interpolation for a given MMSI by comparing the expected
    number of points in the resampled data vs. the actual number.
    """
    try:
        # Load original and resampled data for the given MMSI
        original_files = [f for f in original_dir.rglob("*.parquet") if f.is_file()]
        resampled_files = [f for f in resampled_dir.rglob("*.parquet") if f.is_file()]

        original_df = pl.scan_parquet(original_files).filter(pl.col('mmsi') == mmsi).collect()
        resampled_df = pl.scan_parquet(resampled_files).filter(pl.col('mmsi') == mmsi).collect()

        if original_df.is_empty():
            logging.error(f"No original data found for MMSI {mmsi}")
            return

        logging.info(f"Analyzing MMSI: {mmsi}")

        # Analyze each track for the vessel
        for track_id in original_df['track_id'].unique():
            logging.info(f"--- Track ID: {track_id} ---")
            
            original_track = original_df.filter(pl.col('track_id') == track_id)
            resampled_track = resampled_df.filter(pl.col('track_id') == track_id)

            if original_track.height < 2:
                logging.info("Original track has less than 2 points. Skipping.")
                continue

            # Calculate duration and expected points
            min_ts = original_track['timestamp'].min()
            max_ts = original_track['timestamp'].max()
            duration_seconds = (max_ts - min_ts).total_seconds()
            
            # Expected points = (duration / interval) + 1 (for the start point)
            expected_points = (duration_seconds / 600) + 1
            
            logging.info(f"  Original track duration: {duration_seconds / 3600:.2f} hours")
            logging.info(f"  Original point count: {original_track.height}")
            logging.info(f"  Resampled point count: {resampled_track.height}")
            logging.info(f"  Expected resampled count (approx): {int(expected_points)}")

            # Check if the actual count is reasonably close to the expected count
            # We allow a difference of 1 due to boundary conditions of the 10-min interval
            if abs(resampled_track.height - expected_points) > 1:
                logging.warning("  -> Discrepancy found! The number of resampled points is unexpected.")
            else:
                logging.info("  -> Interpolation count appears correct.")

    except Exception as e:
        logging.error(f"An error occurred during analysis: {e}")

def main(args):
    analyze_mmsi_interpolation(
        args.mmsi,
        Path(args.original_path),
        Path(args.resampled_path)
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(
        description="Verify the interpolation point count for a specific MMSI."
    )
    parser.add_argument("--mmsi", type=int, required=True, help="MMSI to analyze.")
    parser.add_argument("--original_path", type=str, required=True, help="Path to original cleaned data.")
    parser.add_argument("--resampled_path", type=str, required=True, help="Path to resampled data.")
    
    args = parser.parse_args()
    main(args) 