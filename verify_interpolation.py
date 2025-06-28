import pandas as pd
import numpy as np
import os

def main():
    mmsi_to_check = 2182807.0
    original_file = 'data/cleaned_partitioned_ais/year=2025/month=2/day=1/part-0.parquet'
    resampled_file = 'data/resampled_ais_data/year=2025/month/2/day/1/part-0.parquet'

    # Correct the resampled file path
    resampled_file = os.path.join('data', 'resampled_ais_data', 'year=2025', 'month=2', 'day=1', 'part-0.parquet')
    original_file = os.path.join('data', 'cleaned_partitioned_ais', 'year=2025', 'month=2', 'day=1', 'part-0.parquet')


    if not os.path.exists(original_file):
        print(f"Original file not found at: {original_file}")
        return
    if not os.path.exists(resampled_file):
        print(f"Resampled file not found at: {resampled_file}")
        return
        
    df_orig = pd.read_parquet(original_file)
    df_orig = df_orig[df_orig['mmsi'] == mmsi_to_check].copy()
    df_orig['timestamp'] = pd.to_datetime(df_orig['timestamp'])
    df_orig = df_orig.sort_values('timestamp').set_index('timestamp')

    df_resampled = pd.read_parquet(resampled_file)
    df_resampled = df_resampled[df_resampled['mmsi'] == mmsi_to_check].copy()
    df_resampled['timestamp'] = pd.to_datetime(df_resampled['timestamp'])
    df_resampled = df_resampled.sort_values('timestamp').set_index('timestamp')

    interpolated_point_found = False
    for ts in df_resampled.index:
        if ts not in df_orig.index:
            # Check if there are points before and after in the original data
            points_before = df_orig[df_orig.index < ts]
            points_after = df_orig[df_orig.index > ts]

            if not points_before.empty and not points_after.empty:
                interpolated_point = df_resampled.loc[ts]
                p_before = points_before.iloc[-1]
                p_after = points_after.iloc[0]
                interpolated_point_found = True
                break

    if not interpolated_point_found:
        print("Could not find a suitable interpolated point for verification (e.g., at edge of data).")
    else:
        print(f"Found an interpolated point at timestamp: {interpolated_point.name}")
        print(f"  - Interpolated Lat/Lon: ({interpolated_point['lat']:.6f}, {interpolated_point['lon']:.6f})\\n")

        print("Verifying against original data points:")
        print(f"  - Point Before: ts={p_before.name}, lat={p_before['lat']:.6f}, lon={p_before['lon']:.6f}")
        print(f"  - Point After:  ts={p_after.name}, lat={p_after['lat']:.6f}, lon={p_after['lon']:.6f}\\n")

        t1, lat1, lon1 = p_before.name.timestamp(), p_before['lat'], p_before['lon']
        t2, lat2, lon2 = p_after.name.timestamp(), p_after['lat'], p_after['lon']
        t_interp = interpolated_point.name.timestamp()

        ratio = (t_interp - t1) / (t2 - t1)
        calc_lat = lat1 + (lat2 - lat1) * ratio
        calc_lon = lon1 + (lon2 - lon1) * ratio

        print("Manual Linear Interpolation Calculation:")
        print(f"  - Calculated Lat/Lon: ({calc_lat:.6f}, {calc_lon:.6f})")

        lat_match = np.isclose(interpolated_point['lat'], calc_lat)
        lon_match = np.isclose(interpolated_point['lon'], calc_lon)

        print("\\nConfirmation:")
        if lat_match and lon_match:
            print("  - Success! The resampled point's coordinates match the manual linear interpolation.")
        else:
            print("  - Mismatch. The resampled point does not match the manual calculation.")

if __name__ == '__main__':
    main() 