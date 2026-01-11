"""Configuration management for AIS pipeline."""
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Cleaning configuration."""
    track_gap_hours: float = 4.0
    min_track_points: int = 2
    bounds: Dict[str, float] = field(default_factory=lambda: {
        "lat_min": 54.0,
        "lat_max": 58.5,
        "lon_min": 7.0,
        "lon_max": 16.0,
    })
    max_velocity_knots: float = 50.0
    velocity_by_ship_type: Dict[str, float] = field(default_factory=lambda: {
        "cargo": 25.0,
        "tanker": 20.0,
        "passenger": 35.0,
        "fishing": 15.0,
        "default": 50.0,
    })
    collision_distance_threshold_km: float = 50.0
    collision_dbscan_eps_km: float = 5.0
    collision_min_bounce_count: int = 3
    collision_lookback_window: int = 10


@dataclass
class StorageConfig:
    """Storage configuration."""
    s3_bucket: str = ""
    raw_prefix: str = "raw/"
    cleaned_prefix: str = "cleaned/"
    state_prefix: str = "state/"
    local_working_dir: str = "/home/ec2-user/ais-working"


@dataclass
class OutputConfig:
    """Output configuration."""
    format: str = "parquet"
    compression: str = "zstd"
    compression_level: int = 3
    partition_by: list = field(default_factory=lambda: ["year", "month", "day"])
    target_file_size_mb: int = 200
    row_group_size: int = 100_000


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    batch_size_days: int = 7
    checkpoint_interval: int = 1
    num_workers: int = 8


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            PipelineConfig instance
        """
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {path} not found, using defaults")
            data = {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            data = {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            PipelineConfig instance
        """
        config = cls()

        # Parse cleaning config
        cleaning_data = data.get("cleaning", {})
        config.cleaning = CleaningConfig(
            track_gap_hours=cleaning_data.get("track_gap_hours", 4.0),
            min_track_points=cleaning_data.get("min_track_points", 2),
            bounds=cleaning_data.get("bounds", config.cleaning.bounds),
            max_velocity_knots=cleaning_data.get("max_velocity_knots", 50.0),
            velocity_by_ship_type=cleaning_data.get("velocity_by_ship_type", config.cleaning.velocity_by_ship_type),
            collision_distance_threshold_km=cleaning_data.get("collision", {}).get("distance_threshold_km", 50.0),
            collision_dbscan_eps_km=cleaning_data.get("collision", {}).get("dbscan_eps_km", 5.0),
            collision_min_bounce_count=cleaning_data.get("collision", {}).get("min_bounce_count", 3),
            collision_lookback_window=cleaning_data.get("collision", {}).get("lookback_window", 10),
        )

        # Parse storage config
        storage_data = data.get("storage", {})
        config.storage = StorageConfig(
            s3_bucket=storage_data.get("s3_bucket", ""),
            raw_prefix=storage_data.get("raw_prefix", "raw/"),
            cleaned_prefix=storage_data.get("cleaned_prefix", "cleaned/"),
            state_prefix=storage_data.get("state_prefix", "state/"),
            local_working_dir=storage_data.get("local_working_dir", "/home/ec2-user/ais-working"),
        )

        # Parse output config
        output_data = data.get("output", {})
        config.output = OutputConfig(
            format=output_data.get("format", "parquet"),
            compression=output_data.get("compression", "zstd"),
            compression_level=output_data.get("compression_level", 3),
            partition_by=output_data.get("partition_by", ["year", "month", "day"]),
            target_file_size_mb=output_data.get("target_file_size_mb", 200),
            row_group_size=output_data.get("row_group_size", 100_000),
        )

        # Parse processing config
        processing_data = data.get("processing", {})
        config.processing = ProcessingConfig(
            batch_size_days=processing_data.get("batch_size_days", 7),
            checkpoint_interval=processing_data.get("checkpoint_interval", 1),
            num_workers=processing_data.get("num_workers", 8),
        )

        return config


def load_config(path: str = "config/production.yaml") -> PipelineConfig:
    """Load pipeline configuration from file.

    Args:
        path: Path to configuration file

    Returns:
        PipelineConfig instance
    """
    return PipelineConfig.from_yaml(path)
