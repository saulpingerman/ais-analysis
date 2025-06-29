import polars as pl
import argparse
import logging
from pathlib import Path
import shutil
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow as pa

def main(args):
    """
    Reads a dataset partitioned by date and repartitions it by MMSI
    to make vessel-centric queries dramatically faster.
    """
    input_root = Path(args.input)
    output_root = Path(args.output)

    if output_root.exists():
        logging.info(f"Output directory {output_root} already exists. Skipping.")
        return
        
    logging.info(f"Scanning source directory: {input_root}")
    
    # Use pyarrow dataset to handle streaming and partitioning
    source_dataset = ds.dataset(input_root, format="parquet")

    logging.info(f"Repartitioning data to {output_root}. This may take a while...")
    
    # Define the partitioning schema by specifying the fields
    partitioning = ds.partitioning(
        schema=pa.schema([source_dataset.schema.field("mmsi")]),
        flavor="hive"
    )

    # Stream the data and write it to the new partitioned layout
    pq.write_to_dataset(
        source_dataset,
        root_path=output_root,
        partitioning=partitioning,
        existing_data_behavior="overwrite_or_ignore",
        max_partitions=10000
    )

    logging.info("Successfully repartitioned data by MMSI.")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Repartition a dataset by MMSI for faster vessel-based processing."
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Root directory of the source data.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Directory to save the repartitioned data.")
    
    args = parser.parse_args()
    main(args) 