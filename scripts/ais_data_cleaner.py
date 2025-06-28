#!/usr/bin/env python3
"""
AIS Data Cleaner

This module provides `clean_partitioned_ais`, which:
 1) Walks through an AIS dataset partitioned by year/month/day
 2) Deduplicates and bounds-checks positions
 3) Applies stateful speed filtering to remove implausible jumps
 4) Splits voyages into tracks when time gaps exceed a threshold
 5) Writes cleaned Parquet output mirroring the input layout

Usage from bash:

    python ais_cleaner.py \
      --input data/partitioned_ais \
      --output data/cleaned_partitioned_ais \
      --speed_thresh 80.0 \
      --gap_hours 6.0
"""
import argparse
import pathlib
import polars as pl
from math import radians, sin, cos, sqrt, asin

# --- Utility Functions ---

def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance (decimal degrees) in nautical miles.
    """
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)
    a = sin(Δφ/2)**2 + cos(φ1)*cos(φ2)*sin(Δλ/2)**2
    return 2 * 3440.065 * asin(sqrt(a))


def filter_track(df: pl.DataFrame, speed_thresh: float) -> pl.DataFrame:
    """
    Stateful speed filter for a single-vessel DataFrame sorted by 'timestamp'.
    Removes points where speed from last kept fix > threshold.
    """
    cols = df.columns
    kept = {c: [] for c in cols}
    last_lat = last_lon = None
    last_ts = None

    for row in df.rows():
        ts, lat, lon = (
            row[cols.index('timestamp')],
            row[cols.index('lat')],
            row[cols.index('lon')]
        )
        if last_ts is None:
            accept = True
        else:
            dt_h = (ts - last_ts).total_seconds() / 3600.0
            speed = haversine_nm(last_lat, last_lon, lat, lon) / dt_h if dt_h > 0 else float('inf')
            accept = speed <= speed_thresh

        if accept:
            for i, c in enumerate(cols):
                kept[c].append(row[i])
            last_ts, last_lat, last_lon = ts, lat, lon

    return pl.DataFrame(kept, schema=df.schema) if kept[cols[0]] else pl.DataFrame({c: [] for c in cols}, schema=df.schema)

# --- Main Cleaning Function ---

def clean_partitioned_ais(
    input_root: pathlib.Path,
    output_root: pathlib.Path,
    speed_thresh: float = 80.0,
    gap_hours: float = 6.0
) -> None:
    """
    Process AIS data in 'year=*/month=*/day=*' partitions:
      1) Deduplicate & bounds check
      2) Speed filter per vessel
      3) Split voyages into tracks on time gaps > gap_hours
      4) Write cleaned Parquet to output_root

    Args:
        input_root: base directory of source partitions
        output_root: directory for cleaned partitions
        speed_thresh: max allowed speed in knots
        gap_hours: time gap threshold (hours) for splitting tracks
    """
    input_root = pathlib.Path(input_root)
    output_root = pathlib.Path(output_root)
    state: dict[int, tuple] = {}  # mmsi -> (last_timestamp, last_track_id)

    for year_dir in sorted(input_root.glob('year=*')):
        for month_dir in sorted(year_dir.glob('month=*')):
            for day_dir in sorted(month_dir.glob('day=*')):
                files = list(day_dir.glob('*.parquet'))
                if not files:
                    continue
                df = pl.read_parquet([str(f) for f in files])

                # 1) dedupe & bounds
                df = df.unique(subset=['mmsi','timestamp','lat','lon'])
                df = df.filter(
                    pl.col('lat').is_between(-90, 90) &
                    pl.col('lon').is_between(-180, 180)
                )
                df = df.sort(['mmsi','timestamp'])

                cleaned_parts = []
                for m in df['mmsi'].unique().to_list():
                    grp = df.filter(pl.col('mmsi') == m)
                    if grp.is_empty():
                        continue
                    grp = grp.sort('timestamp')

                    last_ts, offset = state.get(m, (None, 0))

                    # 2) speed filter
                    clean = filter_track(grp, speed_thresh)
                    if clean.is_empty():
                        continue

                    # 3) track splitting
                    gap_flag = clean['timestamp'].diff() > pl.duration(hours=gap_hours)
                    clean = clean.with_columns([
                        gap_flag.cast(pl.Int32).alias('gap_int')
                    ])
                    clean = clean.with_columns([
                        (pl.col('gap_int').cum_sum() + offset).alias('track_id')
                    ]).drop('gap_int')

                    # update state
                    new_last_ts = clean['timestamp'].max()
                    raw_max = clean['track_id'].max()
                    new_offset = int(raw_max) if raw_max is not None else offset
                    state[m] = (new_last_ts, new_offset)

                    cleaned_parts.append(clean)

                # concatenate and write
                result = pl.concat(cleaned_parts, how='vertical') if cleaned_parts else pl.DataFrame([], schema=df.schema + ['track_id'])
                out_dir = output_root / year_dir.name / month_dir.name / day_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                result.write_parquet(str(out_dir / 'part-0.parquet'))
                print(f"Processed {year_dir.name}/{month_dir.name}/{day_dir.name}: {result.height} rows")

# --- CLI Entrypoint ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clean AIS partitioned dataset with speed filtering and track splitting.'
    )
    parser.add_argument('--input', '-i', type=pathlib.Path, required=True,
                        help='Input root directory (year=*/month=*/day=*)')
    parser.add_argument('--output', '-o', type=pathlib.Path, required=True,
                        help='Output root directory for cleaned data')
    parser.add_argument('--speed_thresh', '-s', type=float, default=80.0,
                        help='Max vessel speed (knots)')
    parser.add_argument('--gap_hours', '-g', type=float, default=6.0,
                        help='Time gap (hours) to split tracks')
    args = parser.parse_args()
    clean_partitioned_ais(
        input_root=args.input,
        output_root=args.output,
        speed_thresh=args.speed_thresh,
        gap_hours=args.gap_hours
    )
