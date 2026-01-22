"""
Efficient data loader for track-sharded AIS data.

Provides easy loading with filtering by track length, vessel type, etc.
Designed for ML training with sharded data loading.

Example usage:
    from ais_pipeline.io.shard_loader import ShardedAISLoader

    # Initialize loader
    loader = ShardedAISLoader(
        bucket="your-bucket",
        prefix="sharded/",
        min_positions=100,  # Filter short tracks
    )

    # Get shard IDs for parallel workers
    shard_ids = loader.get_shard_ids()

    # Worker 0 loads shards [0, 4, 8, ...]
    # Worker 1 loads shards [1, 5, 9, ...]
    worker_shards = shard_ids[worker_id::num_workers]

    # Load a shard
    for shard_id in worker_shards:
        tracks = loader.load_shard(shard_id)
        for track_df in tracks:
            # track_df is a complete track, sorted by timestamp
            process(track_df)
"""

from typing import Iterator, Optional
import polars as pl


class ShardedAISLoader:
    """Load track-sharded AIS data for ML training."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "sharded/",
        min_positions: Optional[int] = None,
        max_positions: Optional[int] = None,
        min_duration_hours: Optional[float] = None,
        max_duration_hours: Optional[float] = None,
        mmsi_filter: Optional[list[int]] = None,
    ):
        """
        Initialize the loader with filtering criteria.

        Args:
            bucket: S3 bucket name
            prefix: Prefix for sharded data (default: "sharded/")
            min_positions: Minimum positions per track (inclusive)
            max_positions: Maximum positions per track (inclusive)
            min_duration_hours: Minimum track duration in hours
            max_duration_hours: Maximum track duration in hours
            mmsi_filter: List of MMSIs to include (None = all)
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip('/') + '/'
        self.min_positions = min_positions
        self.max_positions = max_positions
        self.min_duration_hours = min_duration_hours
        self.max_duration_hours = max_duration_hours
        self.mmsi_filter = set(mmsi_filter) if mmsi_filter else None

        # Load catalog
        catalog_path = f"s3://{bucket}/{self.prefix}catalog.parquet"
        self.catalog = pl.read_parquet(catalog_path)

        # Apply filters to catalog
        self._apply_filters()

        # Load shard index
        shard_index_path = f"s3://{bucket}/{self.prefix}shard_index.parquet"
        self.shard_index = pl.read_parquet(shard_index_path)

    def _apply_filters(self):
        """Apply filtering criteria to catalog."""
        filters = []

        if self.min_positions is not None:
            filters.append(pl.col("num_positions") >= self.min_positions)

        if self.max_positions is not None:
            filters.append(pl.col("num_positions") <= self.max_positions)

        if self.min_duration_hours is not None:
            filters.append(pl.col("duration_hours") >= self.min_duration_hours)

        if self.max_duration_hours is not None:
            filters.append(pl.col("duration_hours") <= self.max_duration_hours)

        if self.mmsi_filter is not None:
            filters.append(pl.col("mmsi").is_in(self.mmsi_filter))

        if filters:
            combined_filter = filters[0]
            for f in filters[1:]:
                combined_filter = combined_filter & f
            self.catalog = self.catalog.filter(combined_filter)

        # Cache valid track IDs
        self._valid_track_ids = set(self.catalog["track_id"].to_list())

    def get_shard_ids(self) -> list[int]:
        """Get list of shard IDs that contain valid tracks."""
        valid_shards = self.catalog["shard_id"].unique().sort().to_list()
        return valid_shards

    def get_num_tracks(self) -> int:
        """Get total number of tracks matching filter criteria."""
        return self.catalog.height

    def get_track_ids(self) -> list[str]:
        """Get all track IDs matching filter criteria."""
        return self.catalog["track_id"].to_list()

    def get_catalog(self) -> pl.DataFrame:
        """Get the filtered catalog."""
        return self.catalog

    def load_shard(self, shard_id: int) -> Iterator[pl.DataFrame]:
        """
        Load a shard and yield individual tracks.

        Args:
            shard_id: Shard ID to load

        Yields:
            DataFrame for each track, sorted by timestamp
        """
        shard_path = f"s3://{self.bucket}/{self.prefix}shard={shard_id:03d}/tracks.parquet"

        try:
            shard_df = pl.read_parquet(shard_path)
        except Exception as e:
            print(f"Warning: Could not load shard {shard_id}: {e}")
            return

        # Filter to valid tracks only
        shard_df = shard_df.filter(pl.col("track_id").is_in(self._valid_track_ids))

        # Yield individual tracks
        for track_df in shard_df.partition_by("track_id", maintain_order=True):
            yield track_df.sort("timestamp")

    def load_shard_as_dataframe(self, shard_id: int) -> pl.DataFrame:
        """
        Load entire shard as a single DataFrame (filtered).

        Args:
            shard_id: Shard ID to load

        Returns:
            DataFrame with all tracks in the shard (filtered)
        """
        shard_path = f"s3://{self.bucket}/{self.prefix}shard={shard_id:03d}/tracks.parquet"
        shard_df = pl.read_parquet(shard_path)

        # Filter to valid tracks only
        shard_df = shard_df.filter(pl.col("track_id").is_in(self._valid_track_ids))

        return shard_df.sort(["track_id", "timestamp"])

    def load_track(self, track_id: str) -> Optional[pl.DataFrame]:
        """
        Load a specific track by ID.

        Args:
            track_id: Track ID to load

        Returns:
            DataFrame for the track, or None if not found
        """
        # Find shard for this track
        track_info = self.catalog.filter(pl.col("track_id") == track_id)

        if track_info.height == 0:
            return None

        shard_id = track_info["shard_id"][0]
        shard_path = f"s3://{self.bucket}/{self.prefix}shard={shard_id:03d}/tracks.parquet"

        shard_df = pl.read_parquet(shard_path)
        track_df = shard_df.filter(pl.col("track_id") == track_id)

        return track_df.sort("timestamp")

    def __repr__(self):
        return (
            f"ShardedAISLoader("
            f"bucket='{self.bucket}', "
            f"tracks={self.get_num_tracks():,}, "
            f"shards={len(self.get_shard_ids())})"
        )


class ShardedDataset:
    """
    PyTorch-compatible dataset for sharded AIS data.

    Works with DataLoader for parallel loading.
    """

    def __init__(
        self,
        loader: ShardedAISLoader,
        max_seq_len: int = 512,
        features: list[str] = None,
    ):
        """
        Initialize dataset.

        Args:
            loader: ShardedAISLoader instance
            max_seq_len: Maximum sequence length (truncate/pad)
            features: List of feature columns to extract
        """
        self.loader = loader
        self.max_seq_len = max_seq_len
        self.features = features or ["lat", "lon", "sog", "cog", "dt_seconds"]

        # Get all track IDs
        self.track_ids = loader.get_track_ids()

        # Build track_id -> shard_id mapping
        catalog = loader.get_catalog()
        self._track_to_shard = dict(zip(
            catalog["track_id"].to_list(),
            catalog["shard_id"].to_list()
        ))

        # Cache loaded shards
        self._shard_cache = {}

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        import numpy as np

        track_id = self.track_ids[idx]
        shard_id = self._track_to_shard[track_id]

        # Load shard if not cached
        if shard_id not in self._shard_cache:
            self._shard_cache[shard_id] = self.loader.load_shard_as_dataframe(shard_id)

        # Get track data
        shard_df = self._shard_cache[shard_id]
        track_df = shard_df.filter(pl.col("track_id") == track_id).sort("timestamp")

        # Extract features
        features = track_df.select(self.features).to_numpy()

        # Truncate or pad
        if len(features) > self.max_seq_len:
            features = features[:self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            pad_len = self.max_seq_len - len(features)
            mask = np.concatenate([
                np.ones(len(features), dtype=np.float32),
                np.zeros(pad_len, dtype=np.float32)
            ])
            features = np.vstack([
                features,
                np.zeros((pad_len, len(self.features)), dtype=np.float32)
            ])

        return {
            "features": features.astype(np.float32),
            "mask": mask,
            "track_id": track_id,
            "length": min(len(track_df), self.max_seq_len),
        }
