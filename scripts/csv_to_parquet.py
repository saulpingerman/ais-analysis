#!/usr/bin/env python3
"""
Convert DMA AIS ZIP archives in ~/ais_dk/raw → daily Parquet files in ~/ais_dk/parquet.
"""

import pathlib, zipfile, re, contextlib
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

RAW   = pathlib.Path("~/ais_dk/raw").expanduser()
PQOUT = pathlib.Path("~/ais_dk/parquet").expanduser()

NEEDED = {
    "timestamp":   re.compile(r"^#?\s*timestamp$", re.I),
    "mmsi":        re.compile(r"^mmsi$", re.I),
    "latitude":    re.compile(r"^lat", re.I),
    "longitude":   re.compile(r"^lon", re.I),
    "sog":         re.compile(r"^sog$", re.I),
    "cog":         re.compile(r"^cog$", re.I),
    "heading":     re.compile(r"^heading$", re.I),
    "ship_type":   re.compile(r"^ship.?type$", re.I),
}

def canonicalise_cols(raw_cols):
    out = {}
    for canonical, rx in NEEDED.items():
        hits = [c for c in raw_cols if rx.match(c)]
        if not hits:
            raise ValueError(f"Missing required field {canonical!r}")
        out[hits[0]] = canonical
    return out

# ---------- Writer cache ----------------------------------------------------
class WriterPool:
    """
    Keeps one ParquetWriter open per calendar day so we can append row-groups
    without relying on write_table(..., append=True).
    """
    def __init__(self):
        self._pool = {}

    def get(self, out_path: pathlib.Path, schema: pa.Schema, compression="zstd"):
        if out_path not in self._pool:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self._pool[out_path] = pq.ParquetWriter(out_path, schema, compression=compression)
        return self._pool[out_path]

    def close_all(self):
        for writer in self._pool.values():
            with contextlib.suppress(Exception):
                writer.close()
        self._pool.clear()

# ---------- Main ETL --------------------------------------------------------
def process_zip(zpath: pathlib.Path, pool: WriterPool):
    with zipfile.ZipFile(zpath) as zf:
        for member in [m for m in zf.namelist() if m.lower().endswith(".csv")]:
            # ---- learn header names -----------------------------------------
            with zf.open(member) as fhdr:
                header = pd.read_csv(fhdr, nrows=0, sep=None, engine="python").columns
            mapping = canonicalise_cols(header)

            # ---- stream in 1-M row chunks -----------------------------------
            with zf.open(member) as fdata:
                for chunk in pd.read_csv(
                    fdata,
                    usecols=mapping.keys(),
                    parse_dates=[next(k for k in mapping if "Timestamp" in k)],
                    dayfirst=True,
                    sep=None,
                    engine="python",
                    decimal=",",
                    chunksize=1_000_000,
                ):
                    chunk = chunk.rename(columns=mapping)

                    # one Parquet file per calendar day
                    day = chunk.timestamp.dt.strftime("%Y-%m-%d").iloc[0]
                    out_path = PQOUT / day[:4] / day / f"{day}.parquet"

                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    pool.get(out_path, table.schema).write_table(table)

def main():
    print("Scanning raw ZIP archive …")
    pool = WriterPool()
    try:
        for zpath in sorted(RAW.glob("aisdk-*.zip")):
            print(f"⏳  {zpath.name}")
            process_zip(zpath, pool)
    finally:
        pool.close_all()
    print("✅  Conversion finished. Parquet files live in", PQOUT)

if __name__ == "__main__":
    main()
