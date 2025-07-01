#!/usr/bin/env python3
"""
Data Inspection Script for AIS Processing Pipeline

This script helps verify the pipeline's correctness by showing:
1. Raw data samples from ZIP files
2. Processed data samples from Parquet files
3. Before/after comparisons for specific vessels
4. Track continuity analysis across time boundaries
5. Data quality statistics
"""

import argparse
import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
import warnings

import boto3
import pandas as pd
import polars as pl
from tabulate import tabulate

warnings.filterwarnings("ignore")

class AISDataInspector:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
    
    def show_raw_data_sample(self, zip_key: str, num_rows: int = 20):
        """Show sample of raw data from a ZIP file."""
        print(f"\n{'='*60}")
        print(f"RAW DATA SAMPLE: {zip_key}")
        print(f"{'='*60}")
        
        try:
            # Download ZIP file
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=zip_key)
            zip_data = response['Body'].read()
            
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                csv_files = [m for m in zf.infolist() if m.filename.lower().endswith(".csv")]
                
                if not csv_files:
                    print("No CSV files found in ZIP")
                    return
                
                # Take first CSV file
                csv_file = csv_files[0]
                print(f"Reading from: {csv_file.filename}")
                
                with zf.open(csv_file.filename) as csv_stream:
                    # Read as pandas for easier display
                    df = pd.read_csv(csv_stream, nrows=num_rows)
                    
                    print(f"\nShape: {df.shape}")
                    print(f"Columns: {list(df.columns)}")
                    print(f"\nFirst {num_rows} rows:")
                    print(tabulate(df.head(num_rows), headers='keys', tablefmt='grid', showindex=False))
                    
                    # Show data types and basic stats
                    print(f"\nData Types:")
                    print(df.dtypes.to_string())
                    
        except Exception as e:
            print(f"Error reading raw data: {e}")
    
    def show_processed_data_sample(self, parquet_key: str, num_rows: int = 20):
        """Show sample of processed data from a Parquet file."""
        print(f"\n{'='*60}")
        print(f"PROCESSED DATA SAMPLE: {parquet_key}")
        print(f"{'='*60}")
        
        try:
            # Download and read Parquet file
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=parquet_key)
            parquet_data = response['Body'].read()
            
            df = pl.read_parquet(io.BytesIO(parquet_data))
            
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns}")
            
            # Convert to pandas for display
            df_pd = df.head(num_rows).to_pandas()
            print(f"\nFirst {num_rows} rows:")
            print(tabulate(df_pd, headers='keys', tablefmt='grid', showindex=False))
            
            # Show basic statistics
            print(f"\nBasic Statistics:")
            numeric_cols = ['lat', 'lon', 'sog', 'cog', 'heading']
            available_numeric = [col for col in numeric_cols if col in df.columns]
            
            if available_numeric:
                stats_df = df.select(available_numeric).describe()
                print(tabulate(stats_df.to_pandas(), headers='keys', tablefmt='grid'))
            
            # Show unique MMSIs and track counts
            mmsi_count = df.select('mmsi').n_unique()
            track_count = df.select('track_id').n_unique() if 'track_id' in df.columns else 0
            
            print(f"\nData Summary:")
            print(f"  Unique MMSIs: {mmsi_count}")
            print(f"  Unique Tracks: {track_count}")
            print(f"  Time Range: {df.select('timestamp').min().item()} to {df.select('timestamp').max().item()}")
            
        except Exception as e:
            print(f"Error reading processed data: {e}")
    
    def analyze_vessel_tracks(self, parquet_key: str, mmsi: int):
        """Analyze tracks for a specific vessel."""
        print(f"\n{'='*60}")
        print(f"VESSEL TRACK ANALYSIS: MMSI {mmsi}")
        print(f"{'='*60}")
        
        try:
            # Download and read Parquet file
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=parquet_key)
            parquet_data = response['Body'].read()
            
            df = pl.read_parquet(io.BytesIO(parquet_data))
            
            # Filter for specific MMSI
            vessel_df = df.filter(pl.col('mmsi') == mmsi).sort('timestamp')
            
            if vessel_df.height == 0:
                print(f"No data found for MMSI {mmsi}")
                return
            
            print(f"Total records for MMSI {mmsi}: {vessel_df.height}")
            
            # Analyze tracks
            if 'track_id' in vessel_df.columns:
                tracks = vessel_df.group_by('track_id').agg([
                    pl.col('timestamp').min().alias('start_time'),
                    pl.col('timestamp').max().alias('end_time'),
                    pl.col('timestamp').count().alias('num_points'),
                    pl.col('lat').first().alias('start_lat'),
                    pl.col('lon').first().alias('start_lon'),
                    pl.col('lat').last().alias('end_lat'),
                    pl.col('lon').last().alias('end_lon')
                ]).sort('start_time')
                
                print(f"\nTracks found: {tracks.height}")
                print(tabulate(tracks.to_pandas(), headers='keys', tablefmt='grid', showindex=False))
                
                # Check for potential track continuity issues
                tracks_pd = tracks.to_pandas()
                for i in range(len(tracks_pd) - 1):
                    current_end = tracks_pd.iloc[i]['end_time']
                    next_start = tracks_pd.iloc[i + 1]['start_time']
                    gap = (next_start - current_end).total_seconds() / 3600.0
                    
                    print(f"\nGap between track {i+1} and {i+2}: {gap:.2f} hours")
                    if gap < 6.0:  # Less than default gap threshold
                        print(f"  ⚠️  WARNING: Short gap may indicate track split issue")
            
            # Show sample of vessel data
            print(f"\nSample vessel data:")
            sample_df = vessel_df.head(10).to_pandas()
            print(tabulate(sample_df, headers='keys', tablefmt='grid', showindex=False))
            
        except Exception as e:
            print(f"Error analyzing vessel tracks: {e}")
    
    def compare_before_after(self, zip_key: str, parquet_key: str, mmsi: int):
        """Compare data before and after processing for a specific vessel."""
        print(f"\n{'='*60}")
        print(f"BEFORE/AFTER COMPARISON: MMSI {mmsi}")
        print(f"{'='*60}")
        
        # Get raw data
        print("\n--- BEFORE PROCESSING (Raw Data) ---")
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=zip_key)
            zip_data = response['Body'].read()
            
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                csv_files = [m for m in zf.infolist() if m.filename.lower().endswith(".csv")]
                
                raw_data = []
                for csv_file in csv_files[:1]:  # Just first file for speed
                    with zf.open(csv_file.filename) as csv_stream:
                        df = pd.read_csv(csv_stream)
                        
                        # Try to find the MMSI column
                        mmsi_col = None
                        for col in df.columns:
                            if 'mmsi' in col.lower():
                                mmsi_col = col
                                break
                        
                        if mmsi_col and mmsi in df[mmsi_col].values:
                            vessel_raw = df[df[mmsi_col] == mmsi].head(10)
                            print(f"Raw records found: {len(df[df[mmsi_col] == mmsi])}")
                            print(tabulate(vessel_raw, headers='keys', tablefmt='grid', showindex=False))
                            break
                else:
                    print(f"MMSI {mmsi} not found in raw data")
        
        except Exception as e:
            print(f"Error reading raw data: {e}")
        
        # Get processed data
        print("\n--- AFTER PROCESSING (Cleaned Data) ---")
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=parquet_key)
            parquet_data = response['Body'].read()
            
            df = pl.read_parquet(io.BytesIO(parquet_data))
            vessel_df = df.filter(pl.col('mmsi') == mmsi).head(10)
            
            if vessel_df.height > 0:
                print(f"Processed records: {df.filter(pl.col('mmsi') == mmsi).height}")
                print(tabulate(vessel_df.to_pandas(), headers='keys', tablefmt='grid', showindex=False))
            else:
                print(f"MMSI {mmsi} not found in processed data")
        
        except Exception as e:
            print(f"Error reading processed data: {e}")

def main():
    parser = argparse.ArgumentParser(description="Inspect AIS data before and after processing")
    parser.add_argument("--bucket", default="ais-pipeline-data-10179bbf-us-east-1", help="S3 bucket name")
    parser.add_argument("--raw-key", help="S3 key for raw ZIP file")
    parser.add_argument("--processed-key", help="S3 key for processed Parquet file")
    parser.add_argument("--mmsi", type=int, help="Specific MMSI to analyze")
    parser.add_argument("--num-rows", type=int, default=20, help="Number of rows to display")
    
    args = parser.parse_args()
    
    inspector = AISDataInspector(args.bucket)
    
    # Default examples if not specified
    if not args.raw_key:
        args.raw_key = "data/01_raw/ais_dk/raw/aisdk-2025-01-01.zip"
    
    if not args.processed_key:
        args.processed_key = "data/03_primary/test_cleaned_ais/processed_ais_2025-06-30.parquet"
    
    # Show raw data sample
    inspector.show_raw_data_sample(args.raw_key, args.num_rows)
    
    # Show processed data sample
    inspector.show_processed_data_sample(args.processed_key, args.num_rows)
    
    # If MMSI specified, do detailed analysis
    if args.mmsi:
        inspector.analyze_vessel_tracks(args.processed_key, args.mmsi)
        inspector.compare_before_after(args.raw_key, args.processed_key, args.mmsi)
    else:
        # Find a sample MMSI from processed data
        try:
            response = inspector.s3_client.get_object(Bucket=args.bucket, Key=args.processed_key)
            parquet_data = response['Body'].read()
            df = pl.read_parquet(io.BytesIO(parquet_data))
            
            # Get MMSIs with the most records
            mmsi_counts = df.group_by('mmsi').agg(pl.count().alias('count')).sort('count', descending=True)
            top_mmsis = mmsi_counts.head(5)['mmsi'].to_list()
            
            print(f"\n{'='*60}")
            print(f"TOP MMSIs BY RECORD COUNT:")
            print(f"{'='*60}")
            print(tabulate(mmsi_counts.head(10).to_pandas(), headers='keys', tablefmt='grid', showindex=False))
            
            if top_mmsis:
                sample_mmsi = top_mmsis[0]
                print(f"\nUsing MMSI {sample_mmsi} for detailed analysis...")
                inspector.analyze_vessel_tracks(args.processed_key, sample_mmsi)
                inspector.compare_before_after(args.raw_key, args.processed_key, sample_mmsi)
        
        except Exception as e:
            print(f"Error finding sample MMSI: {e}")

if __name__ == "__main__":
    main()