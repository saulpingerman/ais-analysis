"""Command-line interface for AIS pipeline (local filesystem)."""
import json
import logging
from pathlib import Path
from typing import Optional

import polars as pl
import typer

from .config import load_config
from .pipeline import AISPipeline

app = typer.Typer(help="AIS Data Processing Pipeline CLI")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@app.command()
def process(
    config: str = typer.Option("config/production.yaml", "--config", "-c"),
    resume: bool = typer.Option(False, "--resume", "-r"),
    max_files: Optional[int] = typer.Option(None, "--max-files", "-n"),
    output_stats: Optional[str] = typer.Option(None, "--output-stats", "-o"),
):
    """Process AIS data files through the cleaning pipeline."""
    logger.info(f"Loading configuration from {config}")

    try:
        pipeline_config = load_config(config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise typer.Exit(1)

    raw_path = pipeline_config.storage.raw_path
    if not raw_path.exists():
        logger.error(f"Raw directory does not exist: {raw_path}")
        raise typer.Exit(1)

    logger.info(f"Processing data from {raw_path}")

    pipeline = AISPipeline(pipeline_config)
    stats = pipeline.run(resume=resume, max_files=max_files)

    if "error" in stats:
        logger.error(f"Processing failed: {stats['error']}")
        raise typer.Exit(1)

    print("\n" + "=" * 60)
    print("PROCESSING STATISTICS")
    print("=" * 60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)

    if output_stats:
        with open(output_stats, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics written to {output_stats}")


@app.command()
def validate(
    config: str = typer.Option("config/production.yaml", "--config", "-c"),
    input_dir: Optional[str] = typer.Option(None, "--input", "-i",
                                            help="Override clean_dir for validation"),
    report: Optional[str] = typer.Option(None, "--report", "-r"),
):
    """Validate processed AIS data output."""
    pipeline_config = load_config(config)

    target_dir = Path(input_dir) if input_dir else pipeline_config.storage.clean_path
    parquet_files = sorted(target_dir.rglob("*.parquet"))

    if not parquet_files:
        logger.error(f"No parquet files found under {target_dir}")
        raise typer.Exit(1)

    logger.info(f"Found {len(parquet_files)} parquet files")

    validation = {
        "total_files": len(parquet_files),
        "file_sizes_mb": [],
    }

    for path in parquet_files[:10]:
        try:
            file_size_mb = path.stat().st_size / (1024 * 1024)
            validation["file_sizes_mb"].append(file_size_mb)
            if file_size_mb < 10 or file_size_mb > 1000:
                logger.warning(
                    f"File size out of recommended range: {path} ({file_size_mb:.1f} MB)"
                )
        except Exception as e:
            logger.warning(f"Error checking {path}: {e}")

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total parquet files: {validation['total_files']}")
    if validation["file_sizes_mb"]:
        sizes = validation["file_sizes_mb"]
        print(f"Average file size: {sum(sizes) / len(sizes):.1f} MB")
        print(f"Min file size: {min(sizes):.1f} MB")
        print(f"Max file size: {max(sizes):.1f} MB")
    print("=" * 60)

    if report:
        with open(report, "w") as f:
            json.dump(validation, f, indent=2)
        logger.info(f"Report written to {report}")


@app.command()
def stats(
    config: str = typer.Option("config/production.yaml", "--config", "-c"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
):
    """Generate statistics from the processed track catalog."""
    pipeline_config = load_config(config)
    catalog_path = pipeline_config.storage.clean_path / "track_catalog.parquet"

    if not catalog_path.exists():
        logger.error(f"Track catalog not found at {catalog_path}")
        logger.info("Run 'process' first to generate a catalog.")
        raise typer.Exit(1)

    catalog_df = pl.read_parquet(catalog_path)

    result = {
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
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)

    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Stats written to {output}")


def main():
    app()


if __name__ == "__main__":
    main()
