# ais_partitioning.py
#
# Provides utility to load raw AIS Parquet files into DuckDB,
# derive year/month/day columns, and write Hive-style partitions.

import duckdb
import pathlib


def partition_ais(
    raw_pattern: str,
    db_path: str = "data/ais.duckdb",
    out_dir: str = "data/partitioned_ais"
) -> None:
    """
    Reads raw AIS Parquet files into DuckDB, extracts date partitions,
    and writes out Hive-style Parquet partitions by year/month/day.

    Args:
        raw_pattern: Glob pattern pointing to raw Parquet files, e.g.
                     "data/ais_dk/parquet/2025/*/*.parquet".
        db_path:      Path for the DuckDB catalog file.
        out_dir:      Directory under which to write year/month/day partitions.
    """
    # 1) Connect (creates or opens the database)
    con = duckdb.connect(db_path)

    # 2) Build a transient table with extracted date columns
    con.execute(f"""
    CREATE OR REPLACE TABLE ais_raw AS
    SELECT
      mmsi,
      "timestamp",
      latitude AS lat,
      longitude AS lon,
      sog,
      cog,
      heading,
      ship_type,
      EXTRACT(year  FROM "timestamp") AS year,
      EXTRACT(month FROM "timestamp") AS month,
      EXTRACT(day   FROM "timestamp") AS day
    FROM read_parquet('{raw_pattern}');
    """)

    # 3) Ensure output directory exists
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 4) Write Hive-style partitions
    con.execute(f"""
    COPY ais_raw
    TO '{out_dir}/'
    (FORMAT PARQUET, PARTITION_BY (year, month, day));
    """)

    # 5) Verification: list a few written files
    print(f"Partitions written to: {out_dir}")
    for p in sorted(out_path.rglob("year=*/month=*/day=*/part-*.parquet"))[:5]:
        print("  ", p)


# Example usage:
# if __name__ == '__main__':
#     partition_ais(
#         raw_pattern="data/ais_dk/parquet/2025/*/*.parquet",
#         db_path="data/ais.duckdb",
#         out_dir="data/partitioned_ais"
#     )
