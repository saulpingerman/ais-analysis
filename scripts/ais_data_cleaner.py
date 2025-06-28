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
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any
import sys

import polars as pl
from geopy.distance import great_circle

pl.Config.set_tbl_rows(20)


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance in nautical miles using geopy.
    """
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return 0.0
    return great_circle((lat1, lon1), (lat2, lon2)).nautical


def filter_track(df: pl.DataFrame, speed_thresh: float) -> pl.DataFrame:
    """
    Stateful speed filter for a single-vessel DataFrame sorted by 'timestamp'.
    Removes points where speed from last kept fix > threshold.
    """
    if df.is_empty():
        return df

    mask = [False] * df.height
    mask[0] = True

    last_idx = 0
    for i in range(1, df.height):
        last_row = df.row(last_idx, named=True)
        current_row = df.row(i, named=True)

        dt_h = (current_row["timestamp"] - last_row["timestamp"]).total_seconds() / 3600.0
        if dt_h > 0:
            dist_nm = haversine_nm(
                last_row["lat"], last_row["lon"], current_row["lat"], current_row["lon"]
            )
            speed = dist_nm / dt_h
        else:
            speed = float("inf")

        if speed <= speed_thresh:
            mask[i] = True
            last_idx = i

    return df.filter(pl.Series(mask))


def clean_partitioned_ais(
    input_root: Path,
    output_root: Path,
    speed_thresh: float = 80.0,
    gap_hours: float = 6.0,
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
        gap_hours: time gap (hours) to split tracks
    """
    state: dict[
        int, tuple[datetime | None, int]
    ] = {}  # mmsi -> (last_timestamp, last_track_id)
    state_file = Path("track_state.json")
    if state_file.exists():
        with open(state_file, "r") as f:
            state_tuples = json.load(f)
            state = {
                int(k): (datetime.fromisoformat(v[0]) if v[0] else None, v[1])
                for k, v in state_tuples.items()
            }

    for year_dir in sorted(input_root.glob("year=*")):
        for month_dir in sorted(year_dir.glob("month=*")):
            for day_dir in sorted(month_dir.glob("day=*")):
                files = list(day_dir.glob("*.parquet"))
                if not files:
                    continue
                df = pl.read_parquet([str(f) for f in files])

                df = df.sort(["mmsi", "timestamp"])
                df = df.unique(
                    subset=["mmsi", "timestamp", "lat", "lon"], keep="first"
                )
                df = df.filter(
                    pl.col("lat").is_between(-90, 90)
                    & pl.col("lon").is_between(-180, 180)
                )

                cleaned_parts = []
                for m in df["mmsi"].unique().to_list():
                    grp = df.filter(pl.col("mmsi") == m)
                    if grp.is_empty():
                        continue
                    grp = grp.sort("timestamp")

                    last_ts, offset = state.get(m, (None, 0))

                    clean = filter_track(grp, speed_thresh)
                    if clean.is_empty():
                        cleaned_parts.append(clean)
                        continue

                    if clean.height > 0:
                        gap_flag = clean["timestamp"].diff().dt.total_seconds() > (
                            gap_hours * 3600
                        )
                        clean = clean.with_columns(
                            gap_flag.fill_null(False).cast(pl.Int32).alias("gap_int")
                        )
                        clean = clean.with_columns(
                            (pl.col("gap_int").cum_sum() + offset).alias("track_id")
                        ).drop("gap_int")

                        new_last_ts = clean["timestamp"].max()
                        raw_max = clean["track_id"].max()
                        new_offset = int(raw_max) if raw_max is not None else offset
                        state[m] = (new_last_ts, new_offset + 1)
                    else:
                        state[m] = (last_ts, offset)

                    cleaned_parts.append(clean)

                result = (
                    pl.concat(cleaned_parts, how="vertical")
                    if cleaned_parts
                    else pl.DataFrame([], schema={**df.schema, "track_id": pl.Int32})
                )
                out_dir = output_root / year_dir.name / month_dir.name / day_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                result.write_parquet(str(out_dir / "part-0.parquet"))
                logging.info(
                    "Processed %s/%s/%s: %d rows",
                    year_dir.name,
                    month_dir.name,
                    day_dir.name,
                    result.height,
                )

    state_serializable = {
        k: (v[0].isoformat() if v[0] else None, v[1]) for k, v in state.items()
    }
    with open(state_file, "w") as f:
        json.dump(state_serializable, f)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Clean AIS partitioned dataset with speed filtering and track splitting."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Root directory for the input partitioned data (e.g., 'data/01_raw/partitioned_ais').",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Root directory to save the cleaned output data (e.g., 'data/02_intermediate/cleaned_ais').",
    )
    parser.add_argument(
        "--speed_thresh",
        "-s",
        type=float,
        default=80.0,
        help="Maximum plausible vessel speed in knots. Points implying a higher speed from the last valid point will be discarded. (Default: 80.0)",
    )
    parser.add_argument(
        "--gap_hours",
        "-g",
        type=float,
        default=6.0,
        help="Time gap in hours that defines a new track. If the time between two points for a vessel exceeds this, a new track ID will be assigned. (Default: 6.0)",
    )
    args = parser.parse_args()
    clean_partitioned_ais(
        input_root=args.input,
        output_root=args.output,
        speed_thresh=args.speed_thresh,
        gap_hours=args.gap_hours,
    )
