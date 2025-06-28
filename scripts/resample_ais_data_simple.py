import pandas as pd
import os
from tqdm import tqdm

def process_file(source_path, dest_dir):
    """
    Processes a single parquet file by iterating through each ship.
    """
    try:
        df = pd.read_parquet(source_path)
    except Exception as e:
        print(f"Error reading {source_path}: {e}")
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
        
        # --- Enforce Schema from Original File ---
        original_schema = pd.read_parquet(source_path).dtypes
        for col, dtype in original_schema.items():
            if col in final_df.columns:
                final_df[col] = final_df[col].astype(dtype)
        # --- End Schema Enforcement ---

        # Ensure correct column order and save
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(source_path))
        
        final_df = final_df.reindex(columns=original_schema.index) # Reorder columns to match original
        final_df.to_parquet(dest_path)


def main():
    source_base_dir = 'data/cleaned_partitioned_ais'
    dest_base_dir = 'data/resampled_ais_data'

    # Collect all file paths
    all_files = []
    for root, dirs, files in os.walk(source_base_dir):
        if not any(part.startswith('.') for part in root.split(os.sep)):
            for file in files:
                if file.endswith('.parquet'):
                    all_files.append(os.path.join(root, file))

    # Process files with a progress bar
    for source_path in tqdm(all_files, desc="Processing Partitions"):
        relative_path = os.path.relpath(os.path.dirname(source_path), source_base_dir)
        dest_dir = os.path.join(dest_base_dir, relative_path)
        process_file(source_path, dest_dir)

if __name__ == '__main__':
    main() 