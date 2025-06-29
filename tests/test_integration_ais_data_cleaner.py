import shutil
import subprocess
import json
from pathlib import Path
from datetime import datetime, timedelta

import polars as pl
import pytest
import yaml

# Helper function to create test data
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
                "latitude": lat, # Use the raw column names
                "longitude": lon,
                "sog": 10.0,
                "cog": 150.0,
                "heading": 150.0,
                "ship_type": "test_ship"
            }
        )
    return pl.DataFrame(data)

@pytest.fixture
def project_structure(tmp_path):
    """Create a temporary project structure for testing."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    config_path = tmp_path / "config.yaml"

    # Create a test-specific config
    test_config = {
        'speed_thresh': 50.0,
        'gap_hours': 1.0
    }
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)

    yield input_dir, output_dir, config_path
    shutil.rmtree(tmp_path)


def run_cleaner(input_dir: Path, output_dir: Path, config_path: Path):
    """Helper function to run the cleaning script via subprocess."""
    script_path = Path(__file__).parent.parent / "scripts" / "ais_data_cleaner.py"
    
    # We need to change the working directory so the script can find the config file
    process = subprocess.run(
        [
            "python", str(script_path),
            "--input", str(input_dir),
            "--output", str(output_dir),
        ],
        cwd=config_path.parent, # Run from the temp directory
        check=True, capture_output=True, text=True
    )
    return process


def test_cleaner_integration(project_structure):
    """
    Full integration test for the ais_data_cleaner.py script.
    It tests initial processing, speed filtering, track splitting, and state persistence.
    """
    input_dir, output_dir, config_path = project_structure

    # --- Test Case 1: Initial run ---
    day1_dir = input_dir / "year=2024/month=1/day=1"
    day1_dir.mkdir(parents=True)
    
    start_time = datetime(2024, 1, 1, 12, 0, 0)

    # Vessel 111: Tests speed filter. Point 3 is a speed spike.
    vessel1_pts = [
        (55.0, 12.0, 0),    # 12:00 - Kept
        (55.1, 12.1, 30),   # 12:30 - Kept (low speed)
        (85.0, 25.0, 31),   # 12:31 - Speed spike, should be removed
        (55.2, 12.2, 60),   # 13:00 - Kept (speed from 12:30)
    ]
    df1 = create_test_data(111, start_time, vessel1_pts)

    # Vessel 222: Tests gap filter. Point 4 creates a >1hr gap.
    vessel2_pts = [
        (60.0, 15.0, 0),    # 12:00 - Kept
        (60.1, 15.2, 30),   # 12:30 - Kept (low speed)
        (60.2, 15.3, 60),   # 13:00 - Kept
        (60.3, 15.4, 125),  # 14:05 - >1hr gap, should start a new track_id
    ]
    df2 = create_test_data(222, start_time, vessel2_pts)

    test_df_1 = pl.concat([df1, df2])
    test_df_1.write_parquet(day1_dir / "data.parquet")

    run_cleaner(input_dir, output_dir, config_path)

    # --- Verification 1 ---
    output_files_1 = list(output_dir.rglob("*.parquet"))
    assert len(output_files_1) == 1
    result_df_1 = pl.read_parquet(output_files_1[0])

    # 3 points from vessel 111, 4 from vessel 222
    assert result_df_1.height == 7
    v1_res = result_df_1.filter(pl.col("mmsi") == 111)
    assert v1_res.height == 3
    assert v1_res.group_by("track_id").len().height == 1 # Single track

    v2_res = result_df_1.filter(pl.col("mmsi") == 222)
    assert v2_res.height == 4
    assert v2_res.group_by("track_id").len().height == 2 # Should be two tracks
    
    # Check that state file was created correctly
    state_file = output_dir / "track_state.json"
    assert state_file.exists()
    with open(state_file, 'r') as f:
        state = json.load(f)
        assert state['111'] == 0 # Last track ID for vessel 111
        assert state['222'] == 1 # Last track ID for vessel 222

    # --- Test Case 2: State Persistence ---
    day2_dir = input_dir / "year=2024/month=1/day=2"
    day2_dir.mkdir(parents=True)

    # Continue vessel 111 on a new day. It should continue its track.
    vessel1_day2_pts = [(55.3, 12.3, 0)]
    df1_day2 = create_test_data(111, start_time, vessel1_day2_pts)
    df1_day2.write_parquet(day2_dir / "data.parquet")

    run_cleaner(input_dir, output_dir, config_path)

    # --- Verification 2 ---
    output_files_2 = list((output_dir / "year=2024/month=1/day=2").rglob("*.parquet"))
    assert len(output_files_2) == 1
    result_df_2 = pl.read_parquet(output_files_2[0])
    
    # Check that vessel 111 continued on its previous track state
    # The new point should have a track_id based on the previous state's last number (0).
    # Since there's no time gap, it should still be track 0.
    # The track_id itself is a string '111_0'
    assert result_df_2['track_id'][0] == '111_0' 