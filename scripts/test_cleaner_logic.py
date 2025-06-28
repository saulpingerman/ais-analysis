#!/usr/bin/env python3
"""
Test for ais_data_cleaner.py

This script creates a small, controlled dataset to verify the core logic of the
data cleaner, including speed filtering, track splitting, and state persistence.
"""
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl


def run_cleaner(input_dir: Path, output_dir: Path) -> None:
    """Helper function to run the cleaning script via subprocess."""
    script_path = Path(__file__).parent / "ais_data_cleaner.py"
    result = subprocess.run(
        [
            "python",
            str(script_path),
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--speed_thresh",
            "50.0",  # 50 knots
            "--gap_hours",
            "1.0",  # 1 hour
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stderr:
        print("--- Cleaner Stderr ---")
        print(result.stderr)
        print("----------------------")


def create_test_data(
    mmsi: int, start_time: datetime, points: list[tuple[float, float, int]]
) -> pl.DataFrame:
    """Creates a DataFrame for a single vessel track."""
    data = []
    for lat, lon, time_delta_min in points:
        data.append(
            {
                "mmsi": mmsi,
                "timestamp": start_time + timedelta(minutes=time_delta_min),
                "lat": lat,
                "lon": lon,
            }
        )
    return pl.DataFrame(data)


def main():
    """Main test function."""
    # --- Setup test environment ---
    test_root = Path("./temp_test_data")
    input_dir = test_root / "input"
    output_dir = test_root / "output"
    state_file = Path("./track_state.json")

    if test_root.exists():
        shutil.rmtree(test_root)
    if state_file.exists():
        state_file.unlink()

    (input_dir / "year=2024/month=1/day=1").mkdir(parents=True, exist_ok=True)

    # --- Test Case 1: Initial run ---
    print("--- Running Test Case 1: Initial Data Processing ---")
    start_time = datetime(2024, 1, 1, 12, 0, 0)

    # Vessel 111: Tests deduplication and speed filter
    # Point 3 is a massive speed spike and should be removed.
    vessel1_pts = [
        (55.0, 12.0, 0),    # 12:00 - Kept
        (55.0, 12.0, 0),    # 12:00 - Duplicate, should be removed
        (55.1, 12.1, 30),   # 12:30 - Kept (low speed)
        (85.0, 25.0, 31),   # 12:31 - Speed spike, should be removed
        (55.2, 12.2, 60),   # 13:00 - Kept (speed calculated from 12:30)
    ]
    df1 = create_test_data(111, start_time, vessel1_pts)

    # Vessel 222: Tests bounds check and gap filter
    # Point 2 is out of bounds. Point 5 creates a >1hr gap.
    vessel2_pts = [
        (60.0, 15.0, 0),    # 12:00 - Kept
        (100.0, 15.1, 5),   # 12:05 - Out of bounds, should be removed
        (60.1, 15.2, 30),   # 12:30 - Kept (low speed)
        (60.2, 15.3, 60),   # 13:00 - Kept
        (60.3, 15.4, 125),  # 14:05 - >1hr gap, new track_id
    ]
    df2 = create_test_data(222, start_time, vessel2_pts)

    # Vessel 444: Tests two consecutive bad points
    # Points 3 and 4 should be removed due to high speed from point 2
    vessel4_pts = [
        (58.0, 14.0, 0),  # 12:00 - Kept
        (58.1, 14.1, 10), # 12:10 - Kept (speed from P1 is ~41 knots)
        (59.0, 15.0, 11), # 12:11 - Rejected (speed from P2 is huge)
        (59.1, 15.1, 12), # 12:12 - Rejected (speed from P2 is huge)
        (58.2, 14.2, 30), # 12:30 - Kept (speed from P2 is ~20 knots)
    ]
    df4 = create_test_data(444, start_time, vessel4_pts)

    test_df_1 = pl.concat([df1, df2, df4])
    test_file_1 = input_dir / "year=2024/month=1/day=1/part-0.parquet"
    test_df_1.write_parquet(test_file_1)

    run_cleaner(input_dir, output_dir)

    # --- Verification 1 ---
    output_files = list((output_dir).glob("**/*.parquet"))
    assert len(output_files) == 1, "Expected one output file"
    result_df_1 = pl.read_parquet(output_files[0])

    print("Verifying results of Test Case 1...")
    # 3 from vessel 111, 4 from vessel 222, 3 from vessel 444
    expected_rows = 10
    assert (
        result_df_1.height == expected_rows
    ), f"Expected {expected_rows} rows, got {result_df_1.height}"

    v1_res = result_df_1.filter(pl.col("mmsi") == 111)
    assert v1_res.height == 3, f"Vessel 111 should have 3 points, got {v1_res.height}"
    assert v1_res["track_id"].to_list() == [
        0,
        0,
        0,
    ], "Vessel 111 should be on a single track"

    v2_res = result_df_1.filter(pl.col("mmsi") == 222)
    assert v2_res.height == 4, f"Vessel 222 should have 4 points, got {v2_res.height}"
    assert v2_res["track_id"].to_list() == [
        0,
        0,
        0,
        1,
    ], f"Vessel 222 track split is incorrect: {v2_res['track_id'].to_list()}"

    v4_res = result_df_1.filter(pl.col("mmsi") == 444)
    assert v4_res.height == 3, f"Vessel 444 should have 3 points, got {v4_res.height}"
    assert v4_res["track_id"].to_list() == [
        0,
        0,
        0,
    ], "Vessel 444 should be on a single track"
    print("Test Case 1 PASSED.")

    # --- Test Case 2: State Persistence ---
    print("\n--- Running Test Case 2: State Persistence ---")
    shutil.rmtree(input_dir)
    (input_dir / "year=2024/month=1/day=2").mkdir(parents=True, exist_ok=True)

    assert state_file.exists(), "State file was not created"
    with open(state_file, "r") as f:
        state_data = json.load(f)
        assert state_data["111"][1] == 1  # next track_id should be 1
        assert state_data["222"][1] == 2  # next track_id should be 2

    # New data for day 2
    vessel3_pts = [(56.0, 13.0, 0)]  # New vessel
    df3 = create_test_data(333, start_time, vessel3_pts)

    vessel1_day2_pts = [(55.3, 12.3, 0)]  # Continue vessel 1
    df1_day2 = create_test_data(111, start_time, vessel1_day2_pts)

    test_df_2 = pl.concat([df3, df1_day2])
    test_file_2 = input_dir / "year=2024/month=1/day=2/part-0.parquet"
    test_df_2.write_parquet(test_file_2)

    run_cleaner(input_dir, output_dir)

    # --- Verification 2 ---
    output_file_2 = output_dir / "year=2024/month=1/day=2/part-0.parquet"
    assert output_file_2.exists(), "Expected output file for day 2"
    result_df_2 = pl.read_parquet(output_file_2)

    print("Verifying results of Test Case 2...")
    assert result_df_2.height == 2, "Expected 2 rows in second run"

    v1_res_day2 = result_df_2.filter(pl.col("mmsi") == 111)
    assert (
        v1_res_day2["track_id"][0] == 1
    ), f"Vessel 111 should continue on track_id 1, got {v1_res_day2['track_id'][0]}"

    v3_res_day2 = result_df_2.filter(pl.col("mmsi") == 333)
    assert (
        v3_res_day2["track_id"][0] == 0
    ), f"New vessel 333 should start on track_id 0, got {v3_res_day2['track_id'][0]}"

    print("Test Case 2 PASSED.")

    # --- Cleanup ---
    print("\nCleaning up test environment...")
    shutil.rmtree(test_root)
    state_file.unlink()
    print("Test script finished successfully.")


if __name__ == "__main__":
    main() 