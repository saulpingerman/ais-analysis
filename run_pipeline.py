#!/usr/bin/env python3
"""Run the AIS data processing pipeline (local filesystem).

Usage:
    python run_pipeline.py                    # Process all files in raw_dir
    python run_pipeline.py --max-files 2      # Process only 2 files (for testing)
    python run_pipeline.py --resume           # Resume from checkpoint
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ais_pipeline.config import load_config
from ais_pipeline.pipeline import AISPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="AIS Data Processing Pipeline")
    parser.add_argument("--config", "-c", default="config/production.yaml",
                        help="Configuration file path")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--max-files", "-n", type=int, default=None,
                        help="Maximum number of files to process")
    parser.add_argument("--reset", action="store_true",
                        help="Reset state and checkpoint before processing")
    args = parser.parse_args()

    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    raw_path = config.storage.raw_path
    if not raw_path.exists():
        logger.error(f"Raw directory does not exist: {raw_path}")
        sys.exit(1)

    logger.info(f"Raw dir:   {raw_path}")
    logger.info(f"Clean dir: {config.storage.clean_path}")
    logger.info(f"State dir: {config.storage.state_path}")

    pipeline = AISPipeline(config)
    resume = args.resume and not args.reset

    stats = pipeline.run(resume=resume, max_files=args.max_files)

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        sys.exit(1)

    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("=" * 60)


if __name__ == "__main__":
    main()
