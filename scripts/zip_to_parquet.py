#!/usr/bin/env python3
"""
Converts raw AIS DK ZIP archives into partitioned Parquet files using a robust,
single-threaded, memory-efficient approach with pandas.
"""

import io
import pathlib
import re
import zipfile
from collections import Counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# --- Configuration ---
RAW_DATA_DIR = pathlib.Path("~/data/01_raw/ais_dk/").expanduser()
OUTPUT_DIR = pathlib.Path(
    "~/data/02_intermediate/raw_partitioned_ais"
).expanduser()
CHUNK_SIZE = 500_000  # Process CSVs in chunks of 500k rows

# --- Schema Definition ---
NEEDED_COLUMNS = {
    "timestamp": re.compile(r"^#?\s*timestamp$", re.I),
    "mmsi": re.compile(r"^mmsi$", re.I),
    "latitude": re.compile(r"^lat", re.I),
    "longitude": re.compile(r"^lon", re.I),
    "sog": re.compile(r"^sog$", re.I),
    "cog": re.compile(r"^cog$", re.I),
    "heading": re.compile(r"^heading$", re.I),
    "ship_type": re.compile(r"^ship.?type$", re.I),
}


def make_unique_columns(columns: list[str]) -> list[str]:
    """Appends a suffix to duplicate column names to make them unique."""
    counts = Counter(columns)
    if not any(c > 1 for c in counts.values()):
        return columns
    seen = Counter()
    unique_cols = []
    for item in columns:
        if counts[item] > 1:
            seen[item] += 1
            unique_cols.append(f"{item}_{seen[item]}")
        else:
            unique_cols.append(item)
    return unique_cols


def map_source_to_canonical_columns(source_columns: list[str]) -> dict[str, str]:
    """Matches source column names against NEEDED_COLUMNS to create a rename map."""
    rename_map = {}
    for canonical, pattern in NEEDED_COLUMNS.items():
        for source_col in source_columns:
            if pattern.match(source_col):
                if source_col not in rename_map:
                    rename_map[source_col] = canonical
                    break
    # This version is more lenient and won't fail if a column is missing
    return rename_map


def process_zip_archive(zip_path: pathlib.Path, output_path: pathlib.Path):
    """
    Processes a single ZIP archive by streaming its internal CSVs in
    memory-efficient chunks and appending them to a single Parquet file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = None
    try:
        with zipfile.ZipFile(zip_path) as zf:
            csv_members = [m for m in zf.infolist() if m.filename.lower().endswith(".csv")]
            if not csv_members:
                return

            # We no longer set a total, as the in-memory size of data is different
            # from the file size, which makes the percentage misleading.
            # This will show progress rate and total data processed without a percentage.
            desc_text = f"Processing {zip_path.name}"
            with tqdm(desc=desc_text, unit="B", unit_scale=True) as pbar:
                for member in csv_members:
                    with zf.open(member.filename) as csv_stream:
                        text_stream = io.TextIOWrapper(csv_stream, encoding='utf-8', errors='ignore')
                        
                        h1 = text_stream.readline().lstrip("# ").strip()
                        h2 = text_stream.readline().strip()

                        cols_h1 = [c.strip() for c in h1.split(",")]
                        cols_h2 = [c.strip() for c in h2.split(",")]
                        full_source_headers = cols_h1 + (cols_h2[1:] if cols_h2 else [])
                        unique_headers = make_unique_columns(full_source_headers)
                        rename_map = map_source_to_canonical_columns(unique_headers)

                        csv_iterator = pd.read_csv(
                            text_stream,
                            header=None,
                            names=unique_headers,
                            sep=",",
                            on_bad_lines='warn',
                            chunksize=CHUNK_SIZE,
                            low_memory=True
                        )

                        for chunk in csv_iterator:
                            cols_to_keep = [col for col in rename_map.keys() if col in chunk.columns]
                            if not cols_to_keep:
                                pbar.update(chunk.memory_usage(deep=True).sum())
                                continue
                            
                            processed_chunk = chunk[cols_to_keep].rename(columns=rename_map)

                            processed_chunk['timestamp'] = pd.to_datetime(
                                processed_chunk['timestamp'],
                                format='%d/%m/%Y %H:%M:%S',
                                errors='coerce'
                            )
                            processed_chunk = processed_chunk.dropna(subset=['timestamp'])
                            
                            if processed_chunk.empty:
                                pbar.update(chunk.memory_usage(deep=True).sum())
                                continue

                            # Convert to an Arrow Table
                            table = pa.Table.from_pandas(processed_chunk)
                            
                            # Initialize writer with schema from the first valid chunk
                            if writer is None:
                                writer = pq.ParquetWriter(output_path, table.schema, compression='zstd')
                            
                            writer.write_table(table)
                            
                            pbar.update(chunk.memory_usage(deep=True).sum())
    finally:
        if writer is not None:
            writer.close()


def main():
    """
    Main function to discover and process ZIP archives one by one.
    This single-threaded approach is the most robust way to handle
    very large files on memory-constrained systems.
    """
    print(f"Scanning for AIS ZIP archives in {RAW_DATA_DIR}...")
    
    all_zips = list(RAW_DATA_DIR.rglob("aisdk-*.zip"))

    to_process = []
    for zpath in all_zips:
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", zpath.name)
        if not match:
            continue
        
        year, month, day = match.groups()
        out_path = OUTPUT_DIR / f"year={year}" / f"month={month}" / f"day={day}" / "data.parquet"

        if not out_path.exists():
            to_process.append((zpath, out_path))

    if not to_process:
        print("✅ No new archives to process. All up to date.")
        return

    print(f"Found {len(to_process)} new archives to process. Running sequentially.")

    for zip_path, output_path in to_process:
        try:
            process_zip_archive(zip_path, output_path)
        except Exception as e:
            print(f"\n[ERROR] while processing {zip_path.name}: {e}")

    print(f"\n✅ Conversion finished. Parquet files are in {OUTPUT_DIR}")


if __name__ == "__main__":
    main() 