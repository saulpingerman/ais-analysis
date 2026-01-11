"""Cross-file track continuity state management.

Handles maintaining track state across multiple files to ensure:
- Track IDs are consistent for vessels spanning multiple days
- MMSI collision assignments are preserved
- Segment numbers continue correctly
"""
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass
class MMSIState:
    """State for a single MMSI."""
    last_position: Tuple[float, float]  # (lat, lon)
    last_timestamp: str  # ISO format timestamp
    current_segment: int
    cluster_assignment: Optional[str] = None  # "A" or "B" for collision MMSIs


@dataclass
class CollisionRegistryEntry:
    """Registry entry for a detected MMSI collision."""
    detected_date: str
    cluster_a_centroid: Tuple[float, float]
    cluster_b_centroid: Tuple[float, float]


@dataclass
class TrackContinuityState:
    """Complete state for track continuity across files."""
    last_updated: str = ""
    last_file_processed: str = ""
    mmsi_state: Dict[int, MMSIState] = field(default_factory=dict)
    collision_registry: Dict[int, CollisionRegistryEntry] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "last_updated": self.last_updated,
            "last_file_processed": self.last_file_processed,
            "mmsi_state": {
                str(k): {
                    "last_position": list(v.last_position),
                    "last_timestamp": v.last_timestamp,
                    "current_segment": v.current_segment,
                    "cluster_assignment": v.cluster_assignment,
                }
                for k, v in self.mmsi_state.items()
            },
            "collision_registry": {
                str(k): {
                    "detected_date": v.detected_date,
                    "cluster_a_centroid": list(v.cluster_a_centroid),
                    "cluster_b_centroid": list(v.cluster_b_centroid),
                }
                for k, v in self.collision_registry.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackContinuityState":
        """Create from dictionary."""
        state = cls(
            last_updated=data.get("last_updated", ""),
            last_file_processed=data.get("last_file_processed", ""),
        )

        # Parse MMSI state
        for mmsi_str, mmsi_data in data.get("mmsi_state", {}).items():
            mmsi = int(mmsi_str)
            state.mmsi_state[mmsi] = MMSIState(
                last_position=tuple(mmsi_data["last_position"]),
                last_timestamp=mmsi_data["last_timestamp"],
                current_segment=mmsi_data["current_segment"],
                cluster_assignment=mmsi_data.get("cluster_assignment"),
            )

        # Parse collision registry
        for mmsi_str, collision_data in data.get("collision_registry", {}).items():
            mmsi = int(mmsi_str)
            state.collision_registry[mmsi] = CollisionRegistryEntry(
                detected_date=collision_data["detected_date"],
                cluster_a_centroid=tuple(collision_data["cluster_a_centroid"]),
                cluster_b_centroid=tuple(collision_data["cluster_b_centroid"]),
            )

        return state

    def get_starting_segment(self, mmsi: int, first_timestamp: datetime, gap_hours: float) -> int:
        """Get the starting segment number for an MMSI.

        Checks if there's continuity from previous file.

        Args:
            mmsi: Vessel MMSI
            first_timestamp: First timestamp in current file
            gap_hours: Gap threshold in hours

        Returns:
            Starting segment number
        """
        if mmsi not in self.mmsi_state:
            return 0

        mmsi_state = self.mmsi_state[mmsi]
        last_ts = datetime.fromisoformat(mmsi_state.last_timestamp)
        time_gap_hours = (first_timestamp - last_ts).total_seconds() / 3600

        if time_gap_hours <= gap_hours:
            # Continue existing track
            return mmsi_state.current_segment
        else:
            # Start new segment
            return mmsi_state.current_segment + 1

    def get_cluster_assignment(self, mmsi: int) -> Optional[str]:
        """Get cluster assignment for an MMSI if it's a known collision.

        Args:
            mmsi: Vessel MMSI

        Returns:
            "A" or "B" if known collision, None otherwise
        """
        if mmsi in self.mmsi_state and self.mmsi_state[mmsi].cluster_assignment:
            return self.mmsi_state[mmsi].cluster_assignment
        return None

    def is_known_collision(self, mmsi: int) -> bool:
        """Check if MMSI is a known collision.

        Args:
            mmsi: Vessel MMSI

        Returns:
            True if MMSI is in collision registry
        """
        return mmsi in self.collision_registry

    def get_collision_centroids(self, mmsi: int) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get collision centroids for a known collision MMSI.

        Args:
            mmsi: Vessel MMSI

        Returns:
            Tuple of (centroid_a, centroid_b) or None
        """
        if mmsi in self.collision_registry:
            entry = self.collision_registry[mmsi]
            return (entry.cluster_a_centroid, entry.cluster_b_centroid)
        return None

    def update_mmsi_state(
        self,
        mmsi: int,
        last_position: Tuple[float, float],
        last_timestamp: datetime,
        current_segment: int,
        cluster_assignment: Optional[str] = None,
    ):
        """Update state for an MMSI after processing.

        Args:
            mmsi: Vessel MMSI
            last_position: Last (lat, lon)
            last_timestamp: Last timestamp
            current_segment: Current segment number
            cluster_assignment: Cluster assignment if collision
        """
        self.mmsi_state[mmsi] = MMSIState(
            last_position=last_position,
            last_timestamp=last_timestamp.isoformat(),
            current_segment=current_segment,
            cluster_assignment=cluster_assignment,
        )

    def register_collision(
        self,
        mmsi: int,
        centroid_a: Tuple[float, float],
        centroid_b: Tuple[float, float],
        detected_date: str,
    ):
        """Register a new MMSI collision.

        Args:
            mmsi: Vessel MMSI
            centroid_a: Cluster A centroid (lat, lon)
            centroid_b: Cluster B centroid (lat, lon)
            detected_date: Date collision was detected
        """
        self.collision_registry[mmsi] = CollisionRegistryEntry(
            detected_date=detected_date,
            cluster_a_centroid=centroid_a,
            cluster_b_centroid=centroid_b,
        )


def load_state(
    bucket_name: str,
    state_prefix: str = "state/",
    s3_client=None,
) -> TrackContinuityState:
    """Load track continuity state from S3.

    Args:
        bucket_name: S3 bucket name
        state_prefix: Prefix for state files
        s3_client: Optional boto3 S3 client

    Returns:
        TrackContinuityState (empty if no state exists)
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    state_key = f"{state_prefix.rstrip('/')}/track_continuity.json"

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=state_key)
        data = json.loads(response["Body"].read().decode("utf-8"))
        logger.info(f"Loaded state from s3://{bucket_name}/{state_key}")
        return TrackContinuityState.from_dict(data)
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.info("No existing state found, starting fresh")
            return TrackContinuityState()
        else:
            logger.error(f"Error loading state: {e}")
            raise
    except Exception as e:
        logger.error(f"Error loading state: {e}")
        raise


def save_state(
    state: TrackContinuityState,
    bucket_name: str,
    state_prefix: str = "state/",
    s3_client=None,
) -> bool:
    """Save track continuity state to S3.

    Args:
        state: State to save
        bucket_name: S3 bucket name
        state_prefix: Prefix for state files
        s3_client: Optional boto3 S3 client

    Returns:
        True if successful
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    state_key = f"{state_prefix.rstrip('/')}/track_continuity.json"

    try:
        state.last_updated = datetime.utcnow().isoformat()
        state_json = json.dumps(state.to_dict(), indent=2)

        s3_client.put_object(
            Bucket=bucket_name,
            Key=state_key,
            Body=state_json.encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"Saved state to s3://{bucket_name}/{state_key}")
        return True
    except Exception as e:
        logger.error(f"Error saving state: {e}")
        return False
