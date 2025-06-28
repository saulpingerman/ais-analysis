#!/usr/bin/env bash
set -euo pipefail
YEAR=2025
for DAY in 01 02 03 04 05 06 07; do
  FILE=aisdk-${YEAR}-02-${DAY}.zip
  URL=https://web.ais.dk/aisdata/${FILE}
  wget -c "$URL" -P "$AIS_ROOT/raw"
done
