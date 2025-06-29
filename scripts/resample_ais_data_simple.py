import polars as pl
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import shutil

def main(args):
    input_root = Path(args.input)
    output_root = Path(args.output)
    
    if output_root.exists():
        logging.info(f"Removing existing output directory: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    logging.info("Scanning all input files to find unique track IDs...")
    
    # Use Polars to efficiently find all unique track IDs across all files
    all_files = [f for f in input_root.rglob("*.parquet") if f.is_file()]
    if not all_files:
        logging.warning("No parquet files found to process.")
        return
        
    unique_tracks_df = pl.scan_parquet(all_files).select("track_id").unique().collect()
    track_ids = unique_tracks_df["track_id"].to_list()
    
    logging.info(f"Found {len(track_ids)} unique tracks to process.")

    for track_id in tqdm(track_ids, desc="Resampling tracks"):
        try:
            # For each track, scan all files and filter for that track_id
            track_df = pl.scan_parquet(all_files).filter(pl.col("track_id") == track_id).collect()

            if track_df.height < 2:
                continue

            # Perform the resampling on the complete track data
            resampled_track = (
                track_df.sort("timestamp")
                .upsample(time_column="timestamp", every="10m")
                .with_columns(pl.all().forward_fill())
                .with_columns(pl.col(["lat", "lon", "sog", "cog", "heading"]).interpolate())
            )

            # Add a date column for partitioning and write to the output
            if resampled_track.height > 0:
                final_df = resampled_track.with_columns(
                    pl.col("timestamp").dt.date().alias("date")
                )
                final_df.write_parquet(
                    output_root,
                    partition_by="date",
                    pyarrow_options={"compression": "zstd", "compression_level": 3},
                )
        except Exception as e:
            logging.error(f"Failed to process track {track_id}: {e}")

    logging.info("Finished resampling all tracks.")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    parser = argparse.ArgumentParser(
        description="Resample cleaned AIS data to a fixed 10-minute interval using a track-by-track approach."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Root directory for the cleaned, partitioned AIS data.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Root directory to save the resampled output data.",
    )
    args = parser.parse_args()
    main(args) 