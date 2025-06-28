import pandas as pd
import os
import argparse
import logging

def process_file(source_path, dest_dir):
    """
    Processes a single parquet file by iterating through each ship.
    """
    try:
        df = pd.read_parquet(source_path)
    except Exception as e:
        logging.error("Error reading %s: %s", source_path, e)
        return

    # Convert timestamp to datetime objects if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    all_resampled_ships = []
    
    # Iterate over each ship manually
    for mmsi, ship_df in df.groupby('mmsi'):
        if ship_df.shape[0] < 2:
            continue

        ship_df = ship_df.set_index('timestamp').sort_index()

        # Define columns for interpolation vs. forward-filling
        cols_to_interpolate = ['lat', 'lon', 'sog', 'cog', 'heading']
        existing_cols_to_interpolate = [col for col in cols_to_interpolate if col in ship_df.columns]
        
        # Resample and interpolate the specified columns
        interpolated_df = ship_df[existing_cols_to_interpolate].resample('10min').mean().interpolate(method='linear')
        
        # Resample and forward-fill other columns
        other_cols_df = ship_df.drop(columns=existing_cols_to_interpolate).resample('10min').ffill()

        # Combine, fill NaNs, and add mmsi back
        resampled_ship = pd.concat([interpolated_df, other_cols_df], axis=1).ffill().bfill()
        resampled_ship['mmsi'] = mmsi
        
        all_resampled_ships.append(resampled_ship)

    if all_resampled_ships:
        # Combine all processed ships into one dataframe
        final_df = pd.concat(all_resampled_ships).reset_index()
        
        # --- Enforce Schema ---
        # Drop rows where 'track_id' is NaN, which can happen with interpolation
        final_df.dropna(subset=['track_id'], inplace=True)
        
        # Restore original dtypes for non-interpolated columns
        original_dtypes = {k: v for k, v in df.dtypes.items() if k != 'timestamp'}
        for col, dtype in original_dtypes.items():
            if col not in cols_to_interpolate:
                # Use astype with handling for potential mixed types after resampling
                try:
                    final_df[col] = final_df[col].astype(dtype)
                except (ValueError, TypeError):
                    # If direct casting fails, try to apply it element-wise
                    final_df[col] = final_df[col].apply(lambda x: pd.to_numeric(x, errors='coerce')).astype(dtype)
        # --- End Schema Enforcement ---

        # Ensure correct column order and save
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(source_path))
        
        final_df = final_df[df.columns]
        final_df.to_parquet(dest_path)
        logging.info("Successfully processed and saved %s", dest_path)


def main(args):
    source_base_dir = args.input
    dest_base_dir = args.output

    # Collect all file paths
    all_files = []
    for root, dirs, files in os.walk(source_base_dir):
        if not any(part.startswith('.') for part in root.split(os.sep)):
            for file in files:
                if file.endswith('.parquet'):
                    all_files.append(os.path.join(root, file))

    # Process files
    logging.info("Found %d parquet files to process.", len(all_files))
    for source_path in all_files:
        relative_path = os.path.relpath(os.path.dirname(source_path), source_base_dir)
        dest_dir = os.path.join(dest_base_dir, relative_path)
        logging.info("Processing %s -> %s", source_path, dest_dir)
        process_file(source_path, dest_dir)
    logging.info("Finished processing all files.")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Resample cleaned AIS data to a fixed 10-minute interval."
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