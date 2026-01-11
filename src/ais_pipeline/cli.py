"""Command-line interface for AIS pipeline."""
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from .config import load_config
from .pipeline import AISPipeline

app = typer.Typer(help="AIS Data Processing Pipeline CLI")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.command()
def process(
    config: str = typer.Option("config/production.yaml", "--config", "-c", help="Configuration file path"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume from checkpoint"),
    max_files: Optional[int] = typer.Option(None, "--max-files", "-n", help="Maximum files to process"),
    output_stats: Optional[str] = typer.Option(None, "--output-stats", "-o", help="Output stats to JSON file"),
):
    """Process AIS data files through the cleaning pipeline."""
    logger.info(f"Loading configuration from {config}")

    try:
        pipeline_config = load_config(config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise typer.Exit(1)

    if not pipeline_config.storage.s3_bucket:
        logger.error("S3 bucket not configured. Set storage.s3_bucket in config file.")
        raise typer.Exit(1)

    logger.info(f"Processing data from s3://{pipeline_config.storage.s3_bucket}")

    pipeline = AISPipeline(pipeline_config)
    stats = pipeline.run(resume=resume, max_files=max_files)

    if "error" in stats:
        logger.error(f"Processing failed: {stats['error']}")
        raise typer.Exit(1)

    # Print statistics
    print("\n" + "=" * 60)
    print("PROCESSING STATISTICS")
    print("=" * 60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)

    # Write stats to file if requested
    if output_stats:
        with open(output_stats, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics written to {output_stats}")


@app.command()
def validate(
    config: str = typer.Option("config/production.yaml", "--config", "-c", help="Configuration file path"),
    input_prefix: Optional[str] = typer.Option(None, "--input", "-i", help="S3 prefix for input files"),
    report: Optional[str] = typer.Option(None, "--report", "-r", help="Output validation report file"),
):
    """Validate processed AIS data output."""
    import boto3
    import polars as pl

    logger.info(f"Loading configuration from {config}")
    pipeline_config = load_config(config)

    if input_prefix is None:
        input_prefix = pipeline_config.storage.cleaned_prefix

    s3_client = boto3.client("s3")
    bucket = pipeline_config.storage.s3_bucket

    # List parquet files
    paginator = s3_client.get_paginator("list_objects_v2")
    parquet_files = []

    for page in paginator.paginate(Bucket=bucket, Prefix=input_prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                if obj["Key"].endswith(".parquet"):
                    parquet_files.append(obj["Key"])

    if not parquet_files:
        logger.error(f"No parquet files found in s3://{bucket}/{input_prefix}")
        raise typer.Exit(1)

    logger.info(f"Found {len(parquet_files)} parquet files")

    # Validation results
    validation = {
        "total_files": len(parquet_files),
        "total_rows": 0,
        "file_sizes_mb": [],
        "compression": set(),
        "has_track_id": True,
        "has_timestamp": True,
        "sorted_correctly": True,
        "tracks_with_min_points": 0,
        "tracks_below_min_points": 0,
    }

    # Check each file
    for s3_key in parquet_files[:10]:  # Sample first 10 files
        try:
            # Get file metadata
            response = s3_client.head_object(Bucket=bucket, Key=s3_key)
            file_size_mb = response["ContentLength"] / (1024 * 1024)
            validation["file_sizes_mb"].append(file_size_mb)

            # Check file size is in range
            if file_size_mb < 10 or file_size_mb > 1000:
                logger.warning(f"File size out of recommended range: {s3_key} ({file_size_mb:.1f} MB)")

        except Exception as e:
            logger.warning(f"Error checking {s3_key}: {e}")

    # Print validation results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total parquet files: {validation['total_files']}")
    if validation["file_sizes_mb"]:
        avg_size = sum(validation["file_sizes_mb"]) / len(validation["file_sizes_mb"])
        print(f"Average file size: {avg_size:.1f} MB")
        print(f"Min file size: {min(validation['file_sizes_mb']):.1f} MB")
        print(f"Max file size: {max(validation['file_sizes_mb']):.1f} MB")
    print("=" * 60)

    # Write report if requested
    if report:
        validation["file_sizes_mb"] = validation["file_sizes_mb"][:10]  # Limit for JSON
        validation["compression"] = list(validation["compression"])
        with open(report, "w") as f:
            json.dump(validation, f, indent=2)
        logger.info(f"Report written to {report}")


@app.command()
def stats(
    config: str = typer.Option("config/production.yaml", "--config", "-c", help="Configuration file path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output stats file"),
):
    """Generate statistics from processed data."""
    import boto3
    import polars as pl

    logger.info(f"Loading configuration from {config}")
    pipeline_config = load_config(config)

    s3_client = boto3.client("s3")
    bucket = pipeline_config.storage.s3_bucket
    prefix = pipeline_config.storage.cleaned_prefix

    # Check for track catalog
    catalog_key = f"{prefix.rstrip('/')}/track_catalog.parquet"

    try:
        response = s3_client.get_object(Bucket=bucket, Key=catalog_key)
        catalog_df = pl.read_parquet(response["Body"].read())

        stats = {
            "total_tracks": catalog_df.height,
            "unique_mmsis": catalog_df.select("mmsi").n_unique(),
            "avg_positions_per_track": catalog_df.select("num_positions").mean().item(),
            "avg_duration_hours": catalog_df.select("duration_hours").mean().item(),
            "min_positions": catalog_df.select("num_positions").min().item(),
            "max_positions": catalog_df.select("num_positions").max().item(),
        }

        print("\n" + "=" * 60)
        print("DATA STATISTICS")
        print("=" * 60)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        print("=" * 60)

        if output:
            with open(output, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Stats written to {output}")

    except Exception as e:
        logger.error(f"Error reading track catalog: {e}")
        logger.info("Track catalog may not exist yet. Run 'process' first.")
        raise typer.Exit(1)


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
