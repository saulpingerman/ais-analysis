#!/usr/bin/env python3
"""
S3-Based High-Performance AIS Data Processor

This processor provides a complete pipeline for processing AIS data directly from S3:
1. Reads ZIP files from S3 bucket
2. Extracts and processes CSV data in memory
3. Applies vectorized cleaning and speed filtering
4. Creates vessel tracks with proper time-gap detection
5. Maintains track continuity across day boundaries
6. Writes processed data back to S3 in Parquet format

Key features:
- Proper temporal sorting before track creation
- Stateful track ID management across processing batches
- Validation of track continuity and quality
- Memory-efficient processing with correct track boundaries
- Direct S3 integration without local file intermediates
"""

import argparse
import io
import json
import logging
import pickle
import re
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import boto3
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
from geopy.distance import great_circle
from tqdm import tqdm
import yaml

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrackStateManager:
    """Manages track state across processing runs to ensure continuity."""
    
    def __init__(self, s3_client, bucket_name: str, state_prefix: str = "track_state/"):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.state_prefix = state_prefix
        self.mmsi_track_counters = defaultdict(int)
        self.mmsi_last_timestamps = {}
        self.mmsi_last_positions = {}
        
    def load_state(self) -> bool:
        """Load track state from S3."""
        try:
            state_key = f"{self.state_prefix}track_state.pkl"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=state_key)
            state_data = pickle.loads(response['Body'].read())
            
            self.mmsi_track_counters = state_data.get('counters', defaultdict(int))
            self.mmsi_last_timestamps = state_data.get('timestamps', {})
            self.mmsi_last_positions = state_data.get('positions', {})
            
            logger.info(f"Loaded track state for {len(self.mmsi_track_counters)} MMSIs")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.info("No existing track state found, starting fresh")
                return True
            else:
                logger.error(f"Error loading track state: {e}")
                return False
        except Exception as e:
            logger.error(f"Error loading track state: {e}")
            return False
    
    def save_state(self) -> bool:
        """Save track state to S3."""
        try:
            state_data = {
                'counters': dict(self.mmsi_track_counters),
                'timestamps': self.mmsi_last_timestamps,
                'positions': self.mmsi_last_positions
            }
            
            state_key = f"{self.state_prefix}track_state.pkl"
            state_bytes = pickle.dumps(state_data)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=state_key,
                Body=state_bytes
            )
            
            logger.info(f"Saved track state for {len(self.mmsi_track_counters)} MMSIs")
            return True
            
        except Exception as e:
            logger.error(f"Error saving track state: {e}")
            return False
    
    def get_next_track_id(self, mmsi: int, timestamp: datetime, lat: float, lon: float, gap_hours: float) -> str:
        """Get the next track ID for an MMSI, checking for continuity."""
        # Check if this is a continuation of an existing track
        if mmsi in self.mmsi_last_timestamps:
            last_timestamp = self.mmsi_last_timestamps[mmsi]
            time_gap_hours = (timestamp - last_timestamp).total_seconds() / 3600.0
            
            if time_gap_hours <= gap_hours:
                # Continue existing track
                current_track_id = f"{mmsi}_{self.mmsi_track_counters[mmsi]}"
            else:
                # Start new track
                self.mmsi_track_counters[mmsi] += 1
                current_track_id = f"{mmsi}_{self.mmsi_track_counters[mmsi]}"
        else:
            # First track for this MMSI
            current_track_id = f"{mmsi}_{self.mmsi_track_counters[mmsi]}"
        
        # Update state
        self.mmsi_last_timestamps[mmsi] = timestamp
        self.mmsi_last_positions[mmsi] = (lat, lon)
        
        return current_track_id


class S3AISProcessor:
    """S3-based AIS data processor with correct track handling."""
    
    def __init__(self, bucket_name: str, config: Dict):
        self.bucket_name = bucket_name
        self.config = config
        self.s3_client = boto3.client('s3')
        self.track_manager = TrackStateManager(self.s3_client, bucket_name)
        
        # Processing parameters
        self.speed_thresh = config.get('speed_thresh', 80.0)  # knots
        self.gap_hours = config.get('gap_hours', 6.0)
        self.max_time_gap_s = self.gap_hours * 3600
        self.chunk_size = config.get('chunk_size', 500_000)
        self.interpolate = config.get('interpolate', False)
        self.interpolate_interval = config.get('interpolate_interval', 10)  # minutes
        
        # Column mapping for AIS data
        self.needed_columns = {
            "timestamp": re.compile(r"^#?\s*timestamp$", re.I),
            "mmsi": re.compile(r"^mmsi$", re.I),
            "latitude": re.compile(r"^lat", re.I),
            "longitude": re.compile(r"^lon", re.I),
            "sog": re.compile(r"^sog$", re.I),
            "cog": re.compile(r"^cog$", re.I),
            "heading": re.compile(r"^heading$", re.I),
            "ship_type": re.compile(r"^ship.?type$", re.I),
        }
        
    def list_zip_files(self, prefix: str = "data/01_raw/ais_dk/") -> List[str]:
        """List all ZIP files in the S3 bucket with the given prefix."""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            zip_files = []
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.lower().endswith('.zip') and 'aisdk-' in key:
                            zip_files.append(key)
            
            return sorted(zip_files)
        except ClientError as e:
            logger.error(f"Error listing S3 objects: {e}")
            return []
    
    def make_unique_columns(self, columns: List[str]) -> List[str]:
        """Make column names unique by appending suffixes."""
        counts = Counter(columns)
        if not any(c > 1 for c in counts.values()):
            return columns
        
        seen = Counter()
        unique_cols = []
        for item in columns:
            if counts[item] > 1:
                seen[item] += 1
                unique_cols.append(f"{item}_{seen[item]}")
            else:
                unique_cols.append(item)
        return unique_cols
    
    def map_columns(self, source_columns: List[str]) -> Dict[str, str]:
        """Map source columns to canonical names."""
        rename_map = {}
        for canonical, pattern in self.needed_columns.items():
            for source_col in source_columns:
                if pattern.match(source_col):
                    if source_col not in rename_map:
                        rename_map[source_col] = canonical
                        break
        return rename_map
    
    def apply_speed_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply speed filter using proper haversine distance calculation."""
        if df.height < 2:
            return df
        
        try:
            # Calculate proper haversine distance and speed
            df_with_calcs = df.with_columns([
                # Time difference in seconds
                (pl.col("timestamp") - pl.col("timestamp").shift(1))
                .dt.total_seconds()
                .alias("time_diff_s"),
                
                # Previous coordinates for distance calculation
                pl.col("lat").shift(1).alias("prev_lat"),
                pl.col("lon").shift(1).alias("prev_lon")
            ])
            
            # Calculate haversine distance in nautical miles
            df_with_speed = df_with_calcs.with_columns([
                pl.when(pl.col("time_diff_s") > 0)
                .then(
                    # Simplified but more accurate distance calculation in nautical miles
                    # Using the approximation: 1 degree lat = 60 nm, 1 degree lon = 60 * cos(lat) nm
                    (
                        ((pl.col("lat") - pl.col("prev_lat")) * 60.0).pow(2) + 
                        ((pl.col("lon") - pl.col("prev_lon")) * 60.0 * 
                         ((pl.col("lat") + pl.col("prev_lat")) / 2.0 * 3.14159 / 180.0).cos()).pow(2)
                    ).sqrt()
                )
                .otherwise(0.0)
                .alias("distance_nm")
            ]).with_columns([
                # Speed in knots (nautical miles per hour)
                pl.when((pl.col("time_diff_s") > 0) & (pl.col("distance_nm") > 0))
                .then(pl.col("distance_nm") / (pl.col("time_diff_s") / 3600.0))
                .otherwise(0.0)
                .alias("calculated_speed_knots")
            ])
            
            # Filter points with reasonable speeds (strict threshold)
            filtered_df = df_with_speed.filter(
                (pl.col("calculated_speed_knots") <= self.speed_thresh) | 
                (pl.col("time_diff_s").is_null())  # Keep first point
            ).drop(["time_diff_s", "prev_lat", "prev_lon", "distance_nm", "calculated_speed_knots"])
            
            return filtered_df
            
        except Exception as e:
            logger.warning(f"Speed filtering failed, skipping: {e}")
            return df
    
    def interpolate_to_regular_intervals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Interpolate data to regular intervals (e.g., 10 minutes) for each track."""
        if not self.interpolate or df.height < 2:
            return df
        
        try:
            # Group by track_id and interpolate each track separately
            interpolated_tracks = []
            
            for track_df in df.partition_by("track_id"):
                if track_df.height < 2:
                    interpolated_tracks.append(track_df)
                    continue
                
                # Get track time range
                min_time = track_df.select('timestamp').min().item()
                max_time = track_df.select('timestamp').max().item()
                
                # Create regular time grid
                interval_seconds = self.interpolate_interval * 60
                current_time = min_time
                time_grid = []
                
                while current_time <= max_time:
                    time_grid.append(current_time)
                    current_time = current_time + timedelta(seconds=interval_seconds)
                
                if len(time_grid) < 2:
                    interpolated_tracks.append(track_df)
                    continue
                
                # Create time grid DataFrame
                # Ensure consistent data types
                mmsi_value = int(track_df.select('mmsi').item(0, 0))
                track_id_value = str(track_df.select('track_id').item(0, 0))
                
                time_grid_df = pl.DataFrame({
                    'timestamp': time_grid,
                    'track_id': [track_id_value] * len(time_grid),
                    'mmsi': [mmsi_value] * len(time_grid)
                })
                
                # Convert to pandas for easier interpolation
                # Remove duplicates first to avoid reindex issues
                track_pd = track_df.to_pandas().drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
                time_grid_pd = time_grid_df.to_pandas().set_index('timestamp')
                
                # Interpolate numerical columns
                numerical_cols = ['lat', 'lon', 'sog', 'cog', 'heading']
                available_cols = [col for col in numerical_cols if col in track_pd.columns]
                
                if available_cols:
                    # Reindex to time grid and interpolate
                    interpolated_pd = track_pd[available_cols].reindex(
                        track_pd.index.union(time_grid_pd.index)
                    ).interpolate(method='time').reindex(time_grid_pd.index)
                    
                    # Combine with grid metadata
                    result_pd = time_grid_pd.copy()
                    for col in available_cols:
                        result_pd[col] = interpolated_pd[col].astype('float64')  # Ensure float type
                    
                    # Fill non-numerical columns with most common values
                    for col in ['ship_type', 'date']:
                        if col in track_pd.columns:
                            most_common = track_pd[col].mode()
                            if len(most_common) > 0:
                                result_pd[col] = most_common.iloc[0]
                    
                    # Ensure data types are consistent
                    result_pd['mmsi'] = result_pd['mmsi'].astype('int64')
                    result_pd['track_id'] = result_pd['track_id'].astype('str')
                    
                    # Convert back to Polars
                    result_df = pl.from_pandas(result_pd.reset_index())
                    interpolated_tracks.append(result_df)
                else:
                    interpolated_tracks.append(track_df)
            
            if interpolated_tracks:
                return pl.concat(interpolated_tracks, how="diagonal")
            else:
                return df
                
        except Exception as e:
            logger.warning(f"Interpolation failed, returning original data: {e}")
            return df
    
    def create_tracks_with_state(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create track IDs using persistent state management (optimized)."""
        if df.height == 0:
            return df
        
        # For single MMSI groups, we can use vectorized approach with state check
        mmsi = df.select('mmsi').item(0, 0)  # Get the MMSI (should be same for all rows)
        
        # Check if we need to continue from previous state
        start_track_num = self.track_manager.mmsi_track_counters.get(mmsi, 0)
        last_timestamp = self.track_manager.mmsi_last_timestamps.get(mmsi, None)
        
        # Calculate time gaps
        df_with_gaps = df.with_columns([
            (pl.col("timestamp").diff().dt.total_seconds() > self.max_time_gap_s)
            .fill_null(False)  # First row gets False
            .alias("new_track")
        ])
        
        # Handle continuity from previous processing
        if last_timestamp is not None:
            first_timestamp = df.select('timestamp').item(0, 0)
            if (first_timestamp - last_timestamp).total_seconds() > self.max_time_gap_s:
                # Start new track
                start_track_num += 1
        
        # Create track numbers
        df_with_tracks = df_with_gaps.with_columns([
            pl.col("new_track").cum_sum().alias("track_offset")
        ]).with_columns([
            (pl.lit(start_track_num) + pl.col("track_offset")).alias("track_num")
        ]).with_columns([
            (pl.col("mmsi").cast(pl.Utf8) + "_" + pl.col("track_num").cast(pl.Utf8)).alias("track_id")
        ]).drop(["new_track", "track_offset", "track_num"])
        
        # Update state with last values
        last_row = df_with_tracks.tail(1)
        final_timestamp = last_row.select('timestamp').item(0, 0)
        final_lat = last_row.select('lat').item(0, 0)
        final_lon = last_row.select('lon').item(0, 0)
        final_track_num = int(last_row.select('track_id').item(0, 0).split('_')[1])
        
        self.track_manager.mmsi_track_counters[mmsi] = final_track_num
        self.track_manager.mmsi_last_timestamps[mmsi] = final_timestamp
        self.track_manager.mmsi_last_positions[mmsi] = (final_lat, final_lon)
        
        return df_with_tracks
    
    def validate_tracks(self, df: pl.DataFrame) -> Dict[str, any]:
        """Validate track quality and return statistics."""
        stats = {}
        
        if df.height == 0:
            return {"error": "Empty dataframe"}
        
        # Basic stats
        stats['total_records'] = df.height
        stats['unique_mmsis'] = df.select('mmsi').n_unique()
        stats['unique_tracks'] = df.select('track_id').n_unique()
        stats['date_range'] = {
            'start': str(df.select('timestamp').min().item()),
            'end': str(df.select('timestamp').max().item())
        }
        
        # Track length statistics
        track_lengths = df.group_by('track_id').count().select('count')
        stats['track_lengths'] = {
            'mean': float(track_lengths.mean().item()),
            'min': int(track_lengths.min().item()),
            'max': int(track_lengths.max().item())
        }
        
        # Check for very short tracks (potential issues)
        short_tracks = track_lengths.filter(pl.col('count') == 1).height
        stats['single_point_tracks'] = short_tracks
        stats['single_point_percentage'] = (short_tracks / stats['unique_tracks']) * 100
        
        return stats
    
    def process_zip_from_s3(self, s3_key: str) -> Optional[pl.DataFrame]:
        """Process a single ZIP file from S3."""
        try:
            logger.info(f"Processing {s3_key}")
            
            # Download ZIP file to memory
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            zip_data = response['Body'].read()
            
            all_data = []
            
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                csv_members = [m for m in zf.infolist() if m.filename.lower().endswith(".csv")]
                
                for member in csv_members:
                    with zf.open(member.filename) as csv_stream:
                        text_stream = io.TextIOWrapper(csv_stream, encoding='utf-8', errors='ignore')
                        
                        # Skip the header processing and just read with header detection
                        text_stream.seek(0)
                        
                        try:
                            df = pl.read_csv(
                                text_stream,
                                separator=",",
                                ignore_errors=True,
                                truncate_ragged_lines=True,
                                infer_schema_length=1000
                            )
                            
                            # Create rename map from actual columns
                            rename_map = self.map_columns(df.columns)
                            
                            if not rename_map:
                                continue
                            
                            # Select and rename columns
                            cols_to_keep = [col for col in rename_map.keys() if col in df.columns]
                            if not cols_to_keep:
                                continue
                            
                            df = df.select(cols_to_keep).rename(rename_map)
                            
                            # Parse timestamp - try multiple formats
                            if "timestamp" in df.columns:
                                timestamp_formats = [
                                    "%d/%m/%Y %H:%M:%S",
                                    "%Y-%m-%d %H:%M:%S",
                                    "%d-%m-%Y %H:%M:%S",
                                    "%Y/%m/%d %H:%M:%S"
                                ]
                                
                                parsed_df = None
                                for fmt in timestamp_formats:
                                    try:
                                        parsed_df = df.with_columns([
                                            pl.col("timestamp").str.strptime(
                                                pl.Datetime,
                                                format=fmt,
                                                strict=False
                                            )
                                        ]).filter(pl.col("timestamp").is_not_null())
                                        
                                        if not parsed_df.is_empty():
                                            break
                                    except:
                                        continue
                                
                                if parsed_df is not None and not parsed_df.is_empty():
                                    df = parsed_df
                                else:
                                    continue
                            
                            if not df.is_empty():
                                all_data.append(df)
                                
                        except Exception as e:
                            logger.warning(f"Error processing CSV {member.filename}: {e}")
                            continue
            
            if not all_data:
                return None
            
            # Combine all data
            combined_df = pl.concat(all_data, how="diagonal")
            
            # Basic filtering - check which column names we have
            lat_col = "lat" if "lat" in combined_df.columns else "latitude"
            lon_col = "lon" if "lon" in combined_df.columns else "longitude"
            
            combined_df = combined_df.filter(
                pl.col(lat_col).is_between(-90, 90) & 
                pl.col(lon_col).is_between(-180, 180) &
                pl.col("mmsi").is_not_null()
            ).unique(subset=["timestamp", lat_col, lon_col, "mmsi"], keep="first")
            
            # Standardize column names to lat/lon for consistency
            if lat_col == "latitude":
                combined_df = combined_df.rename({"latitude": "lat"})
            if lon_col == "longitude":
                combined_df = combined_df.rename({"longitude": "lon"})
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error processing {s3_key}: {e}")
            return None
    
    def process_mmsi_group(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Process a single MMSI group with correct sorting and track assignment."""
        if df.is_empty():
            return None
        
        try:
            # CRITICAL FIX: Sort by timestamp first!
            df_sorted = df.sort("timestamp")
            
            # Apply speed filter on sorted data
            filtered_df = self.apply_speed_filter(df_sorted)
            
            if filtered_df.is_empty():
                return None
            
            # Create tracks using persistent state
            tracks_df = self.create_tracks_with_state(filtered_df)
            
            # Apply interpolation if enabled
            if self.interpolate:
                tracks_df = self.interpolate_to_regular_intervals(tracks_df)
            
            # Add date column for partitioning
            tracks_df = tracks_df.with_columns([
                pl.col("timestamp").dt.date().alias("date")
            ])
            
            return tracks_df
            
        except Exception as e:
            logger.error(f"Error processing MMSI group: {e}")
            return None
    
    def write_to_s3_parquet(self, df: pl.DataFrame, output_prefix: str, file_suffix: str = "") -> bool:
        """Write DataFrame to S3 as Parquet."""
        try:
            # Convert to PyArrow table
            table = df.to_arrow()
            
            # Create a temporary file-like object
            buffer = io.BytesIO()
            
            # Write parquet to buffer
            pq.write_table(
                table,
                buffer,
                compression='zstd',
                compression_level=3
            )
            
            # Generate S3 key
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            s3_key = f"{output_prefix.rstrip('/')}/processed_ais_{date_str}{file_suffix}.parquet"
            
            # Upload to S3
            buffer.seek(0)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=buffer.getvalue(),
                ContentType='application/octet-stream'
            )
            
            logger.info(f"Wrote {df.height} rows to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing to S3: {e}")
            return False
    
    def process_batch(self, zip_files: List[str], output_prefix: str) -> Dict[str, any]:
        """Process a batch of ZIP files with correct track handling."""
        
        # Load existing track state
        if not self.track_manager.load_state():
            logger.error("Failed to load track state")
            return {"error": "Failed to load track state"}
        
        all_processed_data = []
        processing_stats = {
            'files_processed': 0,
            'total_records': 0,
            'mmsis_processed': set(),
            'errors': []
        }
        
        for zip_file in zip_files:
            try:
                # Process ZIP file
                raw_data = self.process_zip_from_s3(zip_file)
                
                if raw_data is None:
                    processing_stats['errors'].append(f"Failed to process {zip_file}")
                    continue
                
                processing_stats['files_processed'] += 1
                processing_stats['total_records'] += raw_data.height
                
                # Group by MMSI and process each group individually
                mmsi_groups = raw_data.partition_by("mmsi")
                
                for mmsi_df in mmsi_groups:
                    if mmsi_df.height > 0:
                        mmsi = mmsi_df.select('mmsi').item(0, 0)
                        processing_stats['mmsis_processed'].add(mmsi)
                        
                        processed_df = self.process_mmsi_group(mmsi_df)
                        if processed_df is not None:
                            all_processed_data.append(processed_df)
                
            except Exception as e:
                error_msg = f"Error processing {zip_file}: {e}"
                logger.error(error_msg)
                processing_stats['errors'].append(error_msg)
        
        # Combine all processed data
        if all_processed_data:
            final_df = pl.concat(all_processed_data, how="diagonal")
            
            # Validate and get statistics
            validation_stats = self.validate_tracks(final_df)
            processing_stats.update(validation_stats)
            
            # Write to S3
            success = self.write_to_s3_parquet(final_df, output_prefix)
            processing_stats['write_success'] = success
            
            # Save track state
            state_saved = self.track_manager.save_state()
            processing_stats['state_saved'] = state_saved
        else:
            processing_stats['error'] = "No data to process"
        
        # Convert sets to lists for JSON serialization
        processing_stats['mmsis_processed'] = list(processing_stats['mmsis_processed'])
        
        return processing_stats


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description="S3-based AIS Data Processor")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--input-prefix", default="data/01_raw/ais_dk/", 
                       help="S3 prefix for input ZIP files")
    parser.add_argument("--output-prefix", default="data/03_primary/cleaned_ais/",
                       help="S3 prefix for output Parquet files")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
    parser.add_argument("--reset-state", action="store_true", 
                       help="Reset track state (start fresh)")
    parser.add_argument("--speed-thresh", type=float, default=80.0,
                       help="Maximum plausible speed in knots")
    parser.add_argument("--gap-hours", type=float, default=6.0,
                       help="Minimum time gap for new track in hours")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.speed_thresh:
        config['speed_thresh'] = args.speed_thresh
    if args.gap_hours:
        config['gap_hours'] = args.gap_hours
    
    # Initialize processor
    processor = S3AISProcessor(args.bucket, config)
    
    # Reset state if requested
    if args.reset_state:
        logger.info("Resetting track state...")
        processor.track_manager.mmsi_track_counters.clear()
        processor.track_manager.mmsi_last_timestamps.clear()
        processor.track_manager.mmsi_last_positions.clear()
    
    # Get list of ZIP files
    zip_files = processor.list_zip_files(args.input_prefix)
    
    if not zip_files:
        logger.error("No ZIP files found in S3 bucket")
        return
    
    if args.max_files:
        zip_files = zip_files[:args.max_files]
    
    logger.info(f"Found {len(zip_files)} ZIP files to process")
    
    # Process files
    start_time = time.time()
    stats = processor.process_batch(zip_files, args.output_prefix)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
    
    # Print statistics
    print("\n" + "="*60)
    print("PROCESSING STATISTICS")
    print("="*60)
    for key, value in stats.items():
        if key != 'errors':
            print(f"{key}: {value}")
    
    if stats.get('errors'):
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats['errors']:
            print(f"  - {error}")


if __name__ == "__main__":
    main()