"""Processing checkpoint management for resumable processing."""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass
class ProcessingCheckpoint:
    """Checkpoint for resumable processing."""
    last_processed_file: str = ""
    processed_files: List[str] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    last_updated: str = ""
    processing_started: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "last_processed_file": self.last_processed_file,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "last_updated": self.last_updated,
            "processing_started": self.processing_started,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingCheckpoint":
        """Create from dictionary."""
        return cls(
            last_processed_file=data.get("last_processed_file", ""),
            processed_files=data.get("processed_files", []),
            failed_files=data.get("failed_files", []),
            last_updated=data.get("last_updated", ""),
            processing_started=data.get("processing_started", ""),
            stats=data.get("stats", {}),
        )

    def mark_processed(self, filename: str):
        """Mark a file as successfully processed."""
        if filename not in self.processed_files:
            self.processed_files.append(filename)
        self.last_processed_file = filename
        self.last_updated = datetime.utcnow().isoformat()

    def mark_failed(self, filename: str):
        """Mark a file as failed."""
        if filename not in self.failed_files:
            self.failed_files.append(filename)
        self.last_updated = datetime.utcnow().isoformat()

    def get_pending_files(self, all_files: List[str]) -> List[str]:
        """Get list of files that still need processing.

        Args:
            all_files: List of all files to process

        Returns:
            List of files not yet processed
        """
        processed_set = set(self.processed_files)
        return [f for f in all_files if f not in processed_set]

    def update_stats(self, key: str, value: Any):
        """Update a statistic."""
        self.stats[key] = value


def load_checkpoint(
    bucket_name: str,
    state_prefix: str = "state/",
    s3_client=None,
) -> ProcessingCheckpoint:
    """Load processing checkpoint from S3.

    Args:
        bucket_name: S3 bucket name
        state_prefix: Prefix for state files
        s3_client: Optional boto3 S3 client

    Returns:
        ProcessingCheckpoint (empty if no checkpoint exists)
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    checkpoint_key = f"{state_prefix.rstrip('/')}/processing_checkpoint.json"

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=checkpoint_key)
        data = json.loads(response["Body"].read().decode("utf-8"))
        logger.info(f"Loaded checkpoint from s3://{bucket_name}/{checkpoint_key}")
        return ProcessingCheckpoint.from_dict(data)
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.info("No existing checkpoint found, starting fresh")
            checkpoint = ProcessingCheckpoint()
            checkpoint.processing_started = datetime.utcnow().isoformat()
            return checkpoint
        else:
            logger.error(f"Error loading checkpoint: {e}")
            raise
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise


def save_checkpoint(
    checkpoint: ProcessingCheckpoint,
    bucket_name: str,
    state_prefix: str = "state/",
    s3_client=None,
) -> bool:
    """Save processing checkpoint to S3.

    Args:
        checkpoint: Checkpoint to save
        bucket_name: S3 bucket name
        state_prefix: Prefix for state files
        s3_client: Optional boto3 S3 client

    Returns:
        True if successful
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    checkpoint_key = f"{state_prefix.rstrip('/')}/processing_checkpoint.json"

    try:
        checkpoint.last_updated = datetime.utcnow().isoformat()
        checkpoint_json = json.dumps(checkpoint.to_dict(), indent=2)

        s3_client.put_object(
            Bucket=bucket_name,
            Key=checkpoint_key,
            Body=checkpoint_json.encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"Saved checkpoint to s3://{bucket_name}/{checkpoint_key}")
        return True
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        return False
