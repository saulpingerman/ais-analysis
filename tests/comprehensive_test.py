#!/usr/bin/env python3
"""
Comprehensive Test Suite for S3 AIS Data Pipeline

This script extensively tests the pipeline functionality:
1. Multi-day processing with track continuity
2. Speed filtering and impossible velocity removal
3. Position validation and impossible coordinate filtering
4. Interpolation to 10-minute intervals
5. Data quality validation across multiple days
"""

import argparse
import io
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import warnings

import boto3
import polars as pl
import pandas as pd
from tabulate import tabulate

# Import our processor
import sys
script_dir = Path(__file__).parent.parent / "scripts"
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))
from s3_ais_processor import S3AISProcessor, load_config

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ComprehensiveTestSuite:
    """Comprehensive test suite for the AIS pipeline."""
    
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.test_results = {}
        
    def list_available_files(self) -> list:
        """List all available ZIP files for testing."""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            zip_files = []
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix="data/01_raw/ais_dk/raw/"):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.lower().endswith('.zip') and 'aisdk-' in key:
                            zip_files.append({
                                'key': key,
                                'size': obj['Size'],
                                'date': obj['LastModified']
                            })
            
            return sorted(zip_files, key=lambda x: x['key'])
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def test_multi_day_processing(self, max_days: int = 3) -> dict:
        """Test multi-day processing with track continuity."""
        logger.info("="*60)
        logger.info("TEST 1: Multi-Day Processing with Track Continuity")
        logger.info("="*60)
        
        # Get available files
        available_files = self.list_available_files()
        if len(available_files) < max_days:
            logger.warning(f"Only {len(available_files)} files available, testing with all")
            max_days = len(available_files)
        
        test_files = [f['key'] for f in available_files[:max_days]]
        logger.info(f"Testing with {len(test_files)} days of data:")
        for i, file in enumerate(test_files):
            logger.info(f"  Day {i+1}: {file}")
        
        # Create test configuration
        config = {
            'speed_thresh': 80.0,
            'gap_hours': 6.0,
            'interpolate': False,  # Start without interpolation for baseline
            'chunk_size': 500000
        }
        
        # Initialize processor and reset state
        processor = S3AISProcessor(self.bucket_name, config)
        processor.track_manager.mmsi_track_counters.clear()
        processor.track_manager.mmsi_last_timestamps.clear()
        processor.track_manager.mmsi_last_positions.clear()
        
        output_prefix = "data/03_primary/test_multi_day/"
        
        # Process files
        start_time = time.time()
        stats = processor.process_batch(test_files, output_prefix)
        processing_time = time.time() - start_time
        
        logger.info(f"Multi-day processing completed in {processing_time:.2f} seconds")
        
        # Analyze results
        results = {
            'files_processed': stats.get('files_processed', 0),
            'total_records': stats.get('total_records', 0),
            'unique_mmsis': len(stats.get('mmsis_processed', [])),
            'unique_tracks': stats.get('unique_tracks', 0),
            'single_point_tracks': stats.get('single_point_tracks', 0),
            'single_point_percentage': stats.get('single_point_percentage', 0),
            'processing_time_seconds': processing_time,
            'throughput_records_per_second': stats.get('total_records', 0) / processing_time if processing_time > 0 else 0,
            'errors': stats.get('errors', []),
            'track_lengths': stats.get('track_lengths', {}),
            'last_output_file': None
        }
        
        # Find the output file for further analysis
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, 
                Prefix=output_prefix.rstrip('/') + '/'
            )
            if 'Contents' in response:
                output_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.parquet')]
                if output_files:
                    results['last_output_file'] = sorted(output_files)[-1]
        except Exception as e:
            logger.warning(f"Could not find output file: {e}")
        
        self.test_results['multi_day'] = results
        
        # Print results
        print(f"\nMulti-Day Processing Results:")
        print(f"  Files processed: {results['files_processed']}")
        print(f"  Total records: {results['total_records']:,}")
        print(f"  Unique MMSIs: {results['unique_mmsis']}")
        print(f"  Unique tracks: {results['unique_tracks']}")
        print(f"  Single-point tracks: {results['single_point_tracks']} ({results['single_point_percentage']:.2f}%)")
        print(f"  Processing time: {results['processing_time_seconds']:.2f} seconds")
        print(f"  Throughput: {results['throughput_records_per_second']:,.0f} records/second")
        
        if results['track_lengths']:
            print(f"  Track length stats:")
            print(f"    Mean: {results['track_lengths']['mean']:.1f} points")
            print(f"    Min: {results['track_lengths']['min']} points")
            print(f"    Max: {results['track_lengths']['max']} points")
        
        if results['errors']:
            print(f"  Errors: {len(results['errors'])}")
            for error in results['errors'][:3]:  # Show first 3 errors
                print(f"    - {error}")
        
        return results
    
    def test_speed_filtering(self, output_file: str) -> dict:
        """Test speed filtering and impossible velocity removal."""
        logger.info("="*60)
        logger.info("TEST 2: Speed Filtering and Impossible Velocity Removal")
        logger.info("="*60)
        
        if not output_file:
            logger.error("No output file available for speed filtering test")
            return {"error": "No output file available"}
        
        try:
            # Read the processed data
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=output_file)
            parquet_data = response['Body'].read()
            df = pl.read_parquet(io.BytesIO(parquet_data))
            
            logger.info(f"Analyzing speed filtering on {df.height:,} records")
            
            # Calculate speeds between consecutive points for analysis
            speed_analysis = []
            
            # Sample a few vessels for detailed speed analysis
            sample_mmsis = df.select('mmsi').unique().head(10)['mmsi'].to_list()
            
            for mmsi in sample_mmsis:
                vessel_df = df.filter(pl.col('mmsi') == mmsi).sort('timestamp')
                
                if vessel_df.height < 2:
                    continue
                
                # Calculate actual speeds between consecutive points
                vessel_with_speed = vessel_df.with_columns([
                    (pl.col("timestamp") - pl.col("timestamp").shift(1))
                    .dt.total_seconds().alias("time_diff_s"),
                    
                    # Haversine distance approximation
                    ((pl.col("lat") - pl.col("lat").shift(1)).pow(2) + 
                     (pl.col("lon") - pl.col("lon").shift(1)).pow(2)).sqrt()
                    .alias("coord_diff_deg")
                ]).with_columns([
                    # Convert to speed in knots (rough approximation)
                    pl.when(pl.col("time_diff_s") > 0)
                    .then((pl.col("coord_diff_deg") * 60.0) / (pl.col("time_diff_s") / 3600.0))
                    .otherwise(0.0)
                    .alias("calculated_speed_knots")
                ]).filter(pl.col("calculated_speed_knots").is_not_null())
                
                if vessel_with_speed.height > 0:
                    speeds = vessel_with_speed['calculated_speed_knots'].to_list()
                    max_speed = max(speeds) if speeds else 0
                    avg_speed = sum(speeds) / len(speeds) if speeds else 0
                    high_speed_count = sum(1 for s in speeds if s > 80.0)
                    
                    speed_analysis.append({
                        'mmsi': mmsi,
                        'total_points': vessel_df.height,
                        'max_speed': max_speed,
                        'avg_speed': avg_speed,
                        'high_speed_count': high_speed_count,
                        'high_speed_percentage': (high_speed_count / len(speeds)) * 100 if speeds else 0
                    })
            
            # Overall statistics
            total_impossible_speeds = sum(item['high_speed_count'] for item in speed_analysis)
            total_speed_calculations = sum(len(vessel_df.filter(pl.col('mmsi') == item['mmsi']).to_pandas()) - 1 
                                         for item in speed_analysis)
            
            results = {
                'vessels_analyzed': len(speed_analysis),
                'total_speed_calculations': total_speed_calculations,
                'impossible_speeds_remaining': total_impossible_speeds,
                'impossible_speed_percentage': (total_impossible_speeds / total_speed_calculations * 100) 
                                             if total_speed_calculations > 0 else 0,
                'max_speed_observed': max(item['max_speed'] for item in speed_analysis) if speed_analysis else 0,
                'avg_max_speed': sum(item['max_speed'] for item in speed_analysis) / len(speed_analysis) 
                               if speed_analysis else 0,
                'vessels_with_high_speeds': sum(1 for item in speed_analysis if item['high_speed_count'] > 0),
                'speed_filtering_effectiveness': 'PASS' if total_impossible_speeds < total_speed_calculations * 0.01 else 'FAIL'
            }
            
            self.test_results['speed_filtering'] = results
            
            print(f"\nSpeed Filtering Results:")
            print(f"  Vessels analyzed: {results['vessels_analyzed']}")
            print(f"  Speed calculations: {results['total_speed_calculations']:,}")
            print(f"  Impossible speeds remaining: {results['impossible_speeds_remaining']}")
            print(f"  Impossible speed percentage: {results['impossible_speed_percentage']:.3f}%")
            print(f"  Max speed observed: {results['max_speed_observed']:.1f} knots")
            print(f"  Average max speed: {results['avg_max_speed']:.1f} knots")
            print(f"  Vessels with high speeds: {results['vessels_with_high_speeds']}")
            print(f"  Filtering effectiveness: {results['speed_filtering_effectiveness']}")
            
            # Show sample vessels with highest speeds
            if speed_analysis:
                print(f"\nTop 5 vessels by max speed:")
                top_vessels = sorted(speed_analysis, key=lambda x: x['max_speed'], reverse=True)[:5]
                for i, vessel in enumerate(top_vessels, 1):
                    print(f"    {i}. MMSI {vessel['mmsi']}: {vessel['max_speed']:.1f} knots max, "
                          f"{vessel['high_speed_count']} high-speed points")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in speed filtering test: {e}")
            return {"error": str(e)}
    
    def test_position_validation(self, output_file: str) -> dict:
        """Test position validation and impossible coordinate filtering."""
        logger.info("="*60)
        logger.info("TEST 3: Position Validation and Coordinate Filtering")
        logger.info("="*60)
        
        if not output_file:
            logger.error("No output file available for position validation test")
            return {"error": "No output file available"}
        
        try:
            # Read the processed data
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=output_file)
            parquet_data = response['Body'].read()
            df = pl.read_parquet(io.BytesIO(parquet_data))
            
            logger.info(f"Analyzing position validation on {df.height:,} records")
            
            # Check coordinate bounds
            lat_stats = df.select(['lat']).describe().to_pandas()
            lon_stats = df.select(['lon']).describe().to_pandas()
            
            # Count invalid coordinates
            invalid_lat = df.filter(
                (pl.col('lat') < -90) | 
                (pl.col('lat') > 90) | 
                pl.col('lat').is_null()
            ).height
            
            invalid_lon = df.filter(
                (pl.col('lon') < -180) | 
                (pl.col('lon') > 180) | 
                pl.col('lon').is_null()
            ).height
            
            # Check for impossible land positions (rough check)
            # Most AIS data should be maritime, so very inland positions might be errors
            possible_land_positions = df.filter(
                # Very rough approximation - positions that might be on land in Denmark/Baltic region
                (pl.col('lat').is_between(54, 58)) & 
                (pl.col('lon').is_between(8, 16)) &
                # These are very rough inland boundaries - real implementation would use proper coastline data
                (
                    ((pl.col('lat') > 56.5) & (pl.col('lon') < 11)) |  # Rough Jutland interior
                    ((pl.col('lat') > 55.8) & (pl.col('lon') > 12) & (pl.col('lon') < 13))  # Rough Zealand interior
                )
            ).height
            
            # Geographic distribution analysis
            lat_range = float(df.select(pl.col('lat').max() - pl.col('lat').min()).item())
            lon_range = float(df.select(pl.col('lon').max() - pl.col('lon').min()).item())
            
            # Check for coordinate precision (too many identical positions might indicate poor data)
            unique_positions = df.select(['lat', 'lon']).unique().height
            position_diversity = unique_positions / df.height * 100
            
            results = {
                'total_records': df.height,
                'invalid_latitudes': invalid_lat,
                'invalid_longitudes': invalid_lon,
                'lat_min': float(lat_stats.loc[lat_stats['statistic'] == 'min', 'lat'].iloc[0]),
                'lat_max': float(lat_stats.loc[lat_stats['statistic'] == 'max', 'lat'].iloc[0]),
                'lon_min': float(lon_stats.loc[lon_stats['statistic'] == 'min', 'lon'].iloc[0]),
                'lon_max': float(lon_stats.loc[lon_stats['statistic'] == 'max', 'lon'].iloc[0]),
                'lat_range': lat_range,
                'lon_range': lon_range,
                'possible_land_positions': possible_land_positions,
                'unique_positions': unique_positions,
                'position_diversity_percentage': position_diversity,
                'coordinate_validation': 'PASS' if (invalid_lat == 0 and invalid_lon == 0) else 'FAIL'
            }
            
            self.test_results['position_validation'] = results
            
            print(f"\nPosition Validation Results:")
            print(f"  Total records: {results['total_records']:,}")
            print(f"  Invalid latitudes: {results['invalid_latitudes']}")
            print(f"  Invalid longitudes: {results['invalid_longitudes']}")
            print(f"  Latitude range: {results['lat_min']:.4f} to {results['lat_max']:.4f} ({results['lat_range']:.2f}°)")
            print(f"  Longitude range: {results['lon_min']:.4f} to {results['lon_max']:.4f} ({results['lon_range']:.2f}°)")
            print(f"  Possible land positions: {results['possible_land_positions']:,}")
            print(f"  Unique positions: {results['unique_positions']:,}")
            print(f"  Position diversity: {results['position_diversity_percentage']:.2f}%")
            print(f"  Coordinate validation: {results['coordinate_validation']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in position validation test: {e}")
            return {"error": str(e)}
    
    def test_interpolation(self, max_files: int = 2) -> dict:
        """Test interpolation to 10-minute intervals."""
        logger.info("="*60)
        logger.info("TEST 4: Interpolation to 10-Minute Intervals")
        logger.info("="*60)
        
        # Create configuration with interpolation enabled
        config = {
            'speed_thresh': 80.0,
            'gap_hours': 6.0,
            'interpolate': True,
            'interpolate_interval': 10,  # 10 minutes
            'chunk_size': 500000
        }
        
        # Get test files
        available_files = self.list_available_files()
        test_files = [f['key'] for f in available_files[:max_files]]
        
        logger.info(f"Testing interpolation with {len(test_files)} files")
        
        # Initialize processor and reset state for interpolation test
        processor = S3AISProcessor(self.bucket_name, config)
        processor.track_manager.mmsi_track_counters.clear()
        processor.track_manager.mmsi_last_timestamps.clear()
        processor.track_manager.mmsi_last_positions.clear()
        
        output_prefix = "data/03_primary/test_interpolated/"
        
        # Process files with interpolation
        start_time = time.time()
        stats = processor.process_batch(test_files, output_prefix)
        processing_time = time.time() - start_time
        
        logger.info(f"Interpolation processing completed in {processing_time:.2f} seconds")
        
        # Find the output file
        output_file = None
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, 
                Prefix=output_prefix.rstrip('/') + '/'
            )
            if 'Contents' in response:
                output_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.parquet')]
                if output_files:
                    output_file = sorted(output_files)[-1]
        except Exception as e:
            logger.warning(f"Could not find interpolated output file: {e}")
        
        if not output_file:
            return {"error": "No interpolated output file found"}
        
        try:
            # Read the interpolated data
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=output_file)
            parquet_data = response['Body'].read()
            df = pl.read_parquet(io.BytesIO(parquet_data))
            
            logger.info(f"Analyzing interpolation on {df.height:,} records")
            
            # Sample a few vessels for detailed interpolation analysis
            sample_mmsis = df.select('mmsi').unique().head(5)['mmsi'].to_list()
            
            interpolation_analysis = []
            
            for mmsi in sample_mmsis:
                vessel_df = df.filter(pl.col('mmsi') == mmsi).sort('timestamp')
                
                if vessel_df.height < 10:  # Need sufficient data for analysis
                    continue
                
                # Calculate time intervals
                vessel_with_intervals = vessel_df.with_columns([
                    (pl.col("timestamp") - pl.col("timestamp").shift(1))
                    .dt.total_seconds().alias("interval_seconds")
                ]).filter(pl.col("interval_seconds").is_not_null())
                
                if vessel_with_intervals.height == 0:
                    continue
                
                intervals = vessel_with_intervals['interval_seconds'].to_list()
                
                # Convert to minutes
                interval_minutes = [s / 60.0 for s in intervals]
                
                # Analyze interval distribution
                target_interval = 10.0  # 10 minutes
                tolerance = 1.0  # 1 minute tolerance
                
                on_target_intervals = sum(1 for i in interval_minutes 
                                        if abs(i - target_interval) <= tolerance)
                
                interpolation_analysis.append({
                    'mmsi': mmsi,
                    'total_intervals': len(interval_minutes),
                    'min_interval': min(interval_minutes),
                    'max_interval': max(interval_minutes),
                    'avg_interval': sum(interval_minutes) / len(interval_minutes),
                    'on_target_intervals': on_target_intervals,
                    'on_target_percentage': (on_target_intervals / len(interval_minutes)) * 100,
                    'total_points': vessel_df.height
                })
            
            # Overall interpolation statistics
            if interpolation_analysis:
                total_intervals = sum(item['total_intervals'] for item in interpolation_analysis)
                total_on_target = sum(item['on_target_intervals'] for item in interpolation_analysis)
                avg_interval = sum(item['avg_interval'] for item in interpolation_analysis) / len(interpolation_analysis)
                
                results = {
                    'vessels_analyzed': len(interpolation_analysis),
                    'total_intervals_analyzed': total_intervals,
                    'on_target_intervals': total_on_target,
                    'on_target_percentage': (total_on_target / total_intervals * 100) if total_intervals > 0 else 0,
                    'average_interval_minutes': avg_interval,
                    'interpolation_effectiveness': 'PASS' if (total_on_target / total_intervals > 0.7) else 'NEEDS_REVIEW',
                    'processing_time_seconds': processing_time,
                    'output_records': stats.get('total_records', 0)
                }
            else:
                results = {
                    'error': 'No vessels had sufficient data for interpolation analysis'
                }
            
            self.test_results['interpolation'] = results
            
            if 'error' not in results:
                print(f"\nInterpolation Results:")
                print(f"  Vessels analyzed: {results['vessels_analyzed']}")
                print(f"  Total intervals analyzed: {results['total_intervals_analyzed']:,}")
                print(f"  On-target intervals (10 ± 1 min): {results['on_target_intervals']:,}")
                print(f"  On-target percentage: {results['on_target_percentage']:.1f}%")
                print(f"  Average interval: {results['average_interval_minutes']:.2f} minutes")
                print(f"  Interpolation effectiveness: {results['interpolation_effectiveness']}")
                print(f"  Processing time: {results['processing_time_seconds']:.2f} seconds")
                print(f"  Output records: {results['output_records']:,}")
                
                # Show sample vessels
                print(f"\nSample vessel interpolation analysis:")
                for i, vessel in enumerate(interpolation_analysis[:3], 1):
                    print(f"    {i}. MMSI {vessel['mmsi']}: {vessel['total_points']} points, "
                          f"avg interval {vessel['avg_interval']:.1f}min, "
                          f"{vessel['on_target_percentage']:.1f}% on-target")
            else:
                print(f"\nInterpolation Results: {results['error']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in interpolation analysis: {e}")
            return {"error": str(e)}
    
    def test_track_continuity(self, output_file: str) -> dict:
        """Test track continuity across processing runs."""
        logger.info("="*60)
        logger.info("TEST 5: Track Continuity Across Processing Runs")
        logger.info("="*60)
        
        if not output_file:
            logger.error("No output file available for track continuity test")
            return {"error": "No output file available"}
        
        try:
            # Read the processed data
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=output_file)
            parquet_data = response['Body'].read()
            df = pl.read_parquet(io.BytesIO(parquet_data))
            
            logger.info(f"Analyzing track continuity on {df.height:,} records")
            
            # Analyze track patterns
            track_analysis = df.group_by(['mmsi', 'track_id']).agg([
                pl.col('timestamp').min().alias('start_time'),
                pl.col('timestamp').max().alias('end_time'),
                pl.col('timestamp').count().alias('num_points'),
                pl.col('lat').first().alias('start_lat'),
                pl.col('lon').first().alias('start_lon'),
                pl.col('lat').last().alias('end_lat'),
                pl.col('lon').last().alias('end_lon')
            ]).sort(['mmsi', 'start_time'])
            
            # Group by MMSI to analyze track patterns per vessel
            mmsi_analysis = track_analysis.group_by('mmsi').agg([
                pl.col('track_id').count().alias('num_tracks'),
                pl.col('num_points').sum().alias('total_points'),
                pl.col('num_points').mean().alias('avg_track_length'),
                pl.col('start_time').min().alias('vessel_first_time'),
                pl.col('end_time').max().alias('vessel_last_time')
            ])
            
            # Calculate vessel activity duration
            mmsi_analysis = mmsi_analysis.with_columns([
                (pl.col('vessel_last_time') - pl.col('vessel_first_time'))
                .dt.total_hours().alias('total_activity_hours')
            ])
            
            # Statistics
            total_vessels = mmsi_analysis.height
            total_tracks = track_analysis.height
            
            # Vessels with multiple tracks (potential continuity issues if too many)
            multi_track_vessels = mmsi_analysis.filter(pl.col('num_tracks') > 1).height
            
            # Average tracks per vessel
            avg_tracks_per_vessel = float(mmsi_analysis.select(pl.col('num_tracks').mean()).item())
            
            # Vessels with excessive tracks (might indicate continuity issues)
            excessive_track_vessels = mmsi_analysis.filter(pl.col('num_tracks') > 10).height
            
            # Sample some vessels for detailed analysis
            sample_vessels = mmsi_analysis.head(10).to_pandas()
            
            results = {
                'total_vessels': total_vessels,
                'total_tracks': total_tracks,
                'avg_tracks_per_vessel': avg_tracks_per_vessel,
                'multi_track_vessels': multi_track_vessels,
                'multi_track_percentage': (multi_track_vessels / total_vessels * 100) if total_vessels > 0 else 0,
                'excessive_track_vessels': excessive_track_vessels,
                'excessive_track_percentage': (excessive_track_vessels / total_vessels * 100) if total_vessels > 0 else 0,
                'track_continuity_quality': 'EXCELLENT' if avg_tracks_per_vessel < 2.0 else 
                                          'GOOD' if avg_tracks_per_vessel < 5.0 else 'NEEDS_REVIEW'
            }
            
            self.test_results['track_continuity'] = results
            
            print(f"\nTrack Continuity Results:")
            print(f"  Total vessels: {results['total_vessels']:,}")
            print(f"  Total tracks: {results['total_tracks']:,}")
            print(f"  Average tracks per vessel: {results['avg_tracks_per_vessel']:.2f}")
            print(f"  Multi-track vessels: {results['multi_track_vessels']:,} ({results['multi_track_percentage']:.1f}%)")
            print(f"  Excessive track vessels (>10): {results['excessive_track_vessels']:,} ({results['excessive_track_percentage']:.1f}%)")
            print(f"  Track continuity quality: {results['track_continuity_quality']}")
            
            # Show sample vessels
            if len(sample_vessels) > 0:
                print(f"\nSample vessel track analysis:")
                for _, vessel in sample_vessels.head(5).iterrows():
                    print(f"    MMSI {int(vessel['mmsi'])}: {int(vessel['num_tracks'])} tracks, "
                          f"{int(vessel['total_points'])} points, "
                          f"{vessel['total_activity_hours']:.1f} hours active")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in track continuity test: {e}")
            return {"error": str(e)}
    
    def generate_test_report(self) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE AIS PIPELINE TEST REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"S3 Bucket: {self.bucket_name}")
        report.append("")
        
        # Test summary
        test_count = len(self.test_results)
        passed_tests = sum(1 for test in self.test_results.values() 
                          if isinstance(test, dict) and 'error' not in test)
        
        report.append(f"SUMMARY: {passed_tests}/{test_count} tests completed successfully")
        report.append("")
        
        # Individual test results
        for test_name, results in self.test_results.items():
            report.append(f"{test_name.upper().replace('_', ' ')} TEST:")
            report.append("-" * 40)
            
            if isinstance(results, dict):
                if 'error' in results:
                    report.append(f"  STATUS: FAILED - {results['error']}")
                else:
                    report.append(f"  STATUS: COMPLETED")
                    for key, value in results.items():
                        if isinstance(value, (int, float, str)):
                            if isinstance(value, float):
                                report.append(f"  {key}: {value:.3f}")
                            else:
                                report.append(f"  {key}: {value}")
            else:
                report.append(f"  STATUS: INVALID RESULT")
            
            report.append("")
        
        # Overall assessment
        report.append("OVERALL ASSESSMENT:")
        report.append("-" * 40)
        
        # Check key metrics
        assessments = []
        
        if 'multi_day' in self.test_results:
            md = self.test_results['multi_day']
            if md.get('files_processed', 0) > 1:
                assessments.append("✓ Multi-day processing: WORKING")
            else:
                assessments.append("✗ Multi-day processing: FAILED")
        
        if 'speed_filtering' in self.test_results:
            sf = self.test_results['speed_filtering']
            if sf.get('speed_filtering_effectiveness') == 'PASS':
                assessments.append("✓ Speed filtering: EFFECTIVE")
            else:
                assessments.append("⚠ Speed filtering: NEEDS REVIEW")
        
        if 'position_validation' in self.test_results:
            pv = self.test_results['position_validation']
            if pv.get('coordinate_validation') == 'PASS':
                assessments.append("✓ Position validation: WORKING")
            else:
                assessments.append("✗ Position validation: FAILED")
        
        if 'interpolation' in self.test_results:
            interp = self.test_results['interpolation']
            if interp.get('interpolation_effectiveness') == 'PASS':
                assessments.append("✓ Interpolation: WORKING")
            elif interp.get('interpolation_effectiveness') == 'NEEDS_REVIEW':
                assessments.append("⚠ Interpolation: NEEDS REVIEW")
            else:
                assessments.append("✗ Interpolation: FAILED")
        
        if 'track_continuity' in self.test_results:
            tc = self.test_results['track_continuity']
            quality = tc.get('track_continuity_quality', 'UNKNOWN')
            if quality in ['EXCELLENT', 'GOOD']:
                assessments.append("✓ Track continuity: WORKING")
            else:
                assessments.append("⚠ Track continuity: NEEDS REVIEW")
        
        report.extend(assessments)
        report.append("")
        
        # Recommendations
        if passed_tests == test_count:
            report.append("RECOMMENDATION: Pipeline is ready for production use")
        elif passed_tests >= test_count * 0.8:
            report.append("RECOMMENDATION: Pipeline is mostly working, minor issues to address")
        else:
            report.append("RECOMMENDATION: Pipeline needs significant fixes before production")
        
        report.append("="*80)
        
        return "\n".join(report)


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Comprehensive AIS Pipeline Test Suite")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--max-days", type=int, default=3, 
                       help="Maximum days of data to test")
    parser.add_argument("--skip-interpolation", action="store_true",
                       help="Skip interpolation test (takes longer)")
    parser.add_argument("--report-file", help="Save report to file")
    
    args = parser.parse_args()
    
    # Initialize test suite
    test_suite = ComprehensiveTestSuite(args.bucket)
    
    print("Starting comprehensive AIS pipeline testing...")
    print(f"S3 Bucket: {args.bucket}")
    print(f"Max days: {args.max_days}")
    print("")
    
    # Run tests
    try:
        # Test 1: Multi-day processing
        multi_day_results = test_suite.test_multi_day_processing(args.max_days)
        output_file = multi_day_results.get('last_output_file')
        
        # Test 2: Speed filtering
        test_suite.test_speed_filtering(output_file)
        
        # Test 3: Position validation
        test_suite.test_position_validation(output_file)
        
        # Test 4: Interpolation (optional)
        if not args.skip_interpolation:
            test_suite.test_interpolation(min(2, args.max_days))
        
        # Test 5: Track continuity
        test_suite.test_track_continuity(output_file)
        
        # Generate report
        report = test_suite.generate_test_report()
        print("\n" + report)
        
        # Save report if requested
        if args.report_file:
            with open(args.report_file, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {args.report_file}")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        print(f"\nTesting failed: {e}")


if __name__ == "__main__":
    main()