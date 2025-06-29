import sys
import os
import polars as pl
import pytest
from datetime import datetime

# Add the script's directory to the Python path to allow importing it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

# Now we can import the function from the script
from ais_data_cleaner import apply_speed_filter_to_group

def test_speed_filter_removes_impossible_point():
    """
    Tests that the apply_speed_filter_to_group function correctly identifies
    and removes a data point that implies an impossible speed.
    """
    # Create a sample DataFrame. 
    # The 3rd point is far away, creating an impossible jump from the 2nd point.
    # The 4th point is close to the 2nd (last good) point, so it should be kept.
    test_data = {
        'timestamp': [
            datetime(2025, 1, 1, 12, 0, 0),
            datetime(2025, 1, 1, 12, 10, 0), # 10 mins later
            datetime(2025, 1, 1, 12, 20, 0), # 10 mins later
            datetime(2025, 1, 1, 12, 30, 0), # 10 mins later
        ],
        'lat': [55.0, 55.1, 58.0, 55.2], # Big jump in latitude to point 3
        'lon': [12.0, 12.1, 15.0, 12.2], # Big jump in longitude to point 3
    }
    df = pl.DataFrame(test_data)

    # Define the speed threshold in knots
    speed_thresh = 80.0

    # Apply the function to be tested
    cleaned_df = apply_speed_filter_to_group(df, speed_thresh)

    # --- Assertions ---
    # The original DataFrame had 4 rows.
    # The function should have removed the 3rd row (index 2).
    # The final DataFrame should have 3 rows.
    assert cleaned_df.height == 3
    
    # Check that the specific "bad" point is gone.
    # The latitudes of the remaining points should be 55.0, 55.1, and 55.2.
    expected_lats = [55.0, 55.1, 55.2]
    assert cleaned_df['lat'].to_list() == expected_lats

def test_speed_filter_keeps_all_good_points():
    """
    Tests that the speed filter does not remove any points when all speeds
    are plausible.
    """
    test_data = {
        'timestamp': [
            datetime(2025, 1, 1, 12, 0, 0),
            datetime(2025, 1, 1, 12, 10, 0),
            datetime(2025, 1, 1, 12, 20, 0),
        ],
        'lat': [55.0, 55.01, 55.02], # Small, consistent changes
        'lon': [12.0, 12.01, 12.02],
    }
    df = pl.DataFrame(test_data)
    speed_thresh = 80.0
    cleaned_df = apply_speed_filter_to_group(df, speed_thresh)
    
    # No points should have been removed
    assert df.height == cleaned_df.height

# Add more tests as needed 