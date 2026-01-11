"""Main AIS data processing pipeline.

Orchestrates the complete cleaning workflow:
1. Basic Validation
2. MMSI Collision Detection
3. Single Outlier Removal
4. Track Segmentation
5. Track Validation

Handles cross-file track continuity and checkpointing.
"""
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import boto3
import polars as pl
from tqdm import tqdm

from .config import PipelineConfig
from .cleaning.validator import validate_positions, get_validation_stats
from .cleaning.outliers import remove_single_outliers
from .cleaning.collision import detect_mmsi_collision, split_collision_tracks
from .cleaning.segmentation import segment_tracks, filter_short_tracks, add_dt_seconds, get_final_segment_number
from .state.continuity import TrackContinuityState, load_state, save_state
from .state.checkpoint import ProcessingCheckpoint, load_checkpoint, save_checkpoint
from .io.reader import read_zip_from_s3, list_raw_files
from .io.writer import write_partitioned_parquet, generate_track_catalog, write_track_catalog

logger = logging.getLogger(__name__)


class AISPipeline:
    """Main AIS data processing pipeline."""

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.s3_client = boto3.client("s3")
        self.state: Optional[TrackContinuityState] = None
        self.checkpoint: Optional[ProcessingCheckpoint] = None
        self.stats = {
            "files_processed": 0,
            "total_records_input": 0,
            "total_records_output": 0,
            "records_removed_validation": 0,
            "records_removed_outliers": 0,
            "collisions_detected": 0,
            "unique_mmsis": set(),
            "unique_tracks": set(),
        }

    def initialize(self, resume: bool = False) -> bool:
        """Initialize pipeline state and checkpoint.

        Args:
            resume: If True, load existing state; if False, start fresh

        Returns:
            True if initialization successful
        """
        try:
            bucket = self.config.storage.s3_bucket
            state_prefix = self.config.storage.state_prefix

            if resume:
                self.state = load_state(bucket, state_prefix, self.s3_client)
                self.checkpoint = load_checkpoint(bucket, state_prefix, self.s3_client)
                logger.info(f"Resumed from checkpoint: {len(self.checkpoint.processed_files)} files already processed")
            else:
                self.state = TrackContinuityState()
                self.checkpoint = ProcessingCheckpoint()
                self.checkpoint.processing_started = datetime.utcnow().isoformat()
                logger.info("Starting fresh processing run")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False

    def process_mmsi_group(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Process a single MMSI group through the cleaning pipeline.

        Args:
            df: DataFrame for a single MMSI, sorted by timestamp

        Returns:
            Cleaned DataFrame with track_id, or None if no valid data
        """
        if df.is_empty():
            return None

        mmsi = df.select("mmsi").item(0, 0)

        # Step 1: Basic Validation (already done at file level, but double-check)
        original_count = df.height

        # Step 2: Check for MMSI Collision
        cluster_assignment = None
        if self.state.is_known_collision(mmsi):
            # Already known collision - apply existing cluster assignments
            centroids = self.state.get_collision_centroids(mmsi)
            if centroids:
                from .cleaning.collision import CollisionInfo
                known_collision_info = CollisionInfo(
                    mmsi=mmsi,
                    cluster_a_centroid=centroids[0],
                    cluster_b_centroid=centroids[1],
                    bounce_count=0,
                    detection_timestamp="",
                )
                df = split_collision_tracks(df, known_collision_info)
                cluster_assignment = df.select("cluster_assignment").item(0, 0)
        else:
            # Check for new collision
            collision_info = detect_mmsi_collision(
                df,
                distance_threshold_km=self.config.cleaning.collision_distance_threshold_km,
                dbscan_eps_km=self.config.cleaning.collision_dbscan_eps_km,
                min_bounce_count=self.config.cleaning.collision_min_bounce_count,
                lookback_window=self.config.cleaning.collision_lookback_window,
            )

            if collision_info:
                logger.info(f"Detected MMSI collision for {mmsi}: {collision_info.bounce_count} bounces")
                self.stats["collisions_detected"] += 1

                # Register collision
                self.state.register_collision(
                    mmsi=mmsi,
                    centroid_a=collision_info.cluster_a_centroid,
                    centroid_b=collision_info.cluster_b_centroid,
                    detected_date=datetime.utcnow().strftime("%Y-%m-%d"),
                )

                # Split tracks
                df = split_collision_tracks(df, collision_info)
                cluster_assignment = df.select("cluster_assignment").item(0, 0)

        # Step 3: Single Outlier Removal
        pre_outlier_count = df.height
        df = remove_single_outliers(
            df,
            max_velocity_knots=self.config.cleaning.max_velocity_knots,
            velocity_by_ship_type=self.config.cleaning.velocity_by_ship_type,
        )
        self.stats["records_removed_outliers"] += pre_outlier_count - df.height

        if df.is_empty():
            return None

        # Step 4: Track Segmentation
        # Get starting segment from state for continuity
        first_timestamp = df.select("timestamp").item(0, 0)
        starting_segment = self.state.get_starting_segment(
            mmsi,
            first_timestamp,
            self.config.cleaning.track_gap_hours,
        )

        df = segment_tracks(
            df,
            gap_hours=self.config.cleaning.track_gap_hours,
            min_track_points=self.config.cleaning.min_track_points,
            starting_segment=starting_segment,
            cluster_assignment=cluster_assignment,
        )

        # Step 5: Filter Short Tracks
        df = filter_short_tracks(df, self.config.cleaning.min_track_points)

        if df.is_empty():
            return None

        # Add dt_seconds column
        df = add_dt_seconds(df)

        # Update state
        last_row = df.tail(1)
        self.state.update_mmsi_state(
            mmsi=mmsi,
            last_position=(
                last_row.select("lat").item(0, 0),
                last_row.select("lon").item(0, 0),
            ),
            last_timestamp=last_row.select("timestamp").item(0, 0),
            current_segment=get_final_segment_number(df),
            cluster_assignment=cluster_assignment,
        )

        return df

    def process_file(self, s3_key: str) -> Optional[pl.DataFrame]:
        """Process a single ZIP file.

        Args:
            s3_key: S3 key for the ZIP file

        Returns:
            Processed DataFrame or None on error
        """
        # Read file
        raw_df = read_zip_from_s3(
            self.config.storage.s3_bucket,
            s3_key,
            self.s3_client,
        )

        if raw_df is None or raw_df.is_empty():
            logger.warning(f"No data read from {s3_key}")
            return None

        self.stats["total_records_input"] += raw_df.height

        # Step 1: Basic Validation
        validated_df = validate_positions(
            raw_df,
            bounds=self.config.cleaning.bounds,
        )

        self.stats["records_removed_validation"] += raw_df.height - validated_df.height

        if validated_df.is_empty():
            logger.warning(f"All records filtered out from {s3_key}")
            return None

        # Sort by MMSI and timestamp for processing
        validated_df = validated_df.sort(["mmsi", "timestamp"])

        # Process by MMSI
        all_processed = []
        mmsi_groups = validated_df.partition_by("mmsi")

        for mmsi_df in mmsi_groups:
            mmsi = mmsi_df.select("mmsi").item(0, 0)
            self.stats["unique_mmsis"].add(mmsi)

            processed_df = self.process_mmsi_group(mmsi_df)
            if processed_df is not None:
                all_processed.append(processed_df)

                # Track unique track IDs
                for track_id in processed_df.select("track_id").unique().to_series().to_list():
                    self.stats["unique_tracks"].add(track_id)

        if not all_processed:
            return None

        # Combine all processed data
        result_df = pl.concat(all_processed, how="diagonal")
        self.stats["total_records_output"] += result_df.height

        return result_df

    def run(self, resume: bool = False, max_files: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete processing pipeline.

        Args:
            resume: If True, resume from checkpoint
            max_files: Optional maximum number of files to process

        Returns:
            Dictionary with processing statistics
        """
        start_time = time.time()

        # Initialize
        if not self.initialize(resume):
            return {"error": "Failed to initialize"}

        # Get list of files to process
        all_files = list_raw_files(
            self.config.storage.s3_bucket,
            self.config.storage.raw_prefix,
            self.s3_client,
        )

        if not all_files:
            return {"error": "No files found to process"}

        # Filter to pending files
        pending_files = self.checkpoint.get_pending_files(all_files)

        if max_files:
            pending_files = pending_files[:max_files]

        logger.info(f"Processing {len(pending_files)} files (of {len(all_files)} total)")

        # Process files
        all_results = []

        for s3_key in tqdm(pending_files, desc="Processing files"):
            try:
                result_df = self.process_file(s3_key)

                if result_df is not None and not result_df.is_empty():
                    all_results.append(result_df)

                # Update checkpoint
                self.checkpoint.mark_processed(s3_key)
                self.stats["files_processed"] += 1

                # Save state and checkpoint periodically
                if self.stats["files_processed"] % self.config.processing.checkpoint_interval == 0:
                    save_state(self.state, self.config.storage.s3_bucket, self.config.storage.state_prefix, self.s3_client)
                    save_checkpoint(self.checkpoint, self.config.storage.s3_bucket, self.config.storage.state_prefix, self.s3_client)

            except Exception as e:
                logger.error(f"Error processing {s3_key}: {e}")
                self.checkpoint.mark_failed(s3_key)

        # Combine all results
        if all_results:
            final_df = pl.concat(all_results, how="diagonal")

            # Write partitioned output
            write_partitioned_parquet(
                final_df,
                self.config.storage.s3_bucket,
                self.config.storage.cleaned_prefix,
                compression=self.config.output.compression,
                compression_level=self.config.output.compression_level,
                row_group_size=self.config.output.row_group_size,
                s3_client=self.s3_client,
            )

            # Generate and write track catalog
            catalog_df = generate_track_catalog(final_df, self.config.storage.cleaned_prefix)
            if not catalog_df.is_empty():
                write_track_catalog(
                    catalog_df,
                    self.config.storage.s3_bucket,
                    self.config.storage.cleaned_prefix,
                    self.s3_client,
                )

        # Save final state
        self.state.last_file_processed = pending_files[-1] if pending_files else ""
        save_state(self.state, self.config.storage.s3_bucket, self.config.storage.state_prefix, self.s3_client)
        save_checkpoint(self.checkpoint, self.config.storage.s3_bucket, self.config.storage.state_prefix, self.s3_client)

        elapsed_time = time.time() - start_time

        # Prepare final stats
        final_stats = {
            "files_processed": self.stats["files_processed"],
            "total_records_input": self.stats["total_records_input"],
            "total_records_output": self.stats["total_records_output"],
            "records_removed_validation": self.stats["records_removed_validation"],
            "records_removed_outliers": self.stats["records_removed_outliers"],
            "collisions_detected": self.stats["collisions_detected"],
            "unique_mmsis": len(self.stats["unique_mmsis"]),
            "unique_tracks": len(self.stats["unique_tracks"]),
            "elapsed_seconds": elapsed_time,
            "records_per_second": self.stats["total_records_input"] / elapsed_time if elapsed_time > 0 else 0,
        }

        logger.info(f"Processing complete in {elapsed_time:.1f} seconds")

        return final_stats
