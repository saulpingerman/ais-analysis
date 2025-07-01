# AIS Data Processing Pipeline

High-performance S3-based pipeline for processing Automatic Identification System (AIS) data from the Danish Maritime Authority.

## Features

- **S3-Native Processing**: Direct ZIP file processing from S3 without local storage
- **Track Continuity**: Maintains vessel track integrity across day boundaries  
- **High Performance**: Vectorized operations using Polars for 10-100x speed improvements
- **Memory Efficient**: Processes datasets larger than available RAM
- **Data Validation**: Built-in quality checks and statistics
- **Configurable**: Flexible parameters for different use cases

## Quick Start

### 1. Automated Setup

```bash
./setup.sh
```

This will install dependencies and guide you through configuration.

### 2. Manual Setup (Alternative)

Install dependencies:
```bash
pip install -r requirements.txt
```

Configure AWS credentials:
```bash
aws configure
```

### 3. Download AIS Data

Download data from Danish Maritime Authority:

```bash
# Download recent data (daily files)
python scripts/download_ais_data.py --start-date 2024-03-01 --end-date 2024-03-31

# Download historical data (monthly files)  
python scripts/download_ais_data.py --start-date 2020-01-01 --end-date 2020-12-31

# Or use the shell wrapper
./scripts/fetch_ais_data.sh 2024-01-01 2024-01-31
```

### 4. Process AIS Data

Process downloaded data from S3:

```bash
python scripts/s3_ais_processor.py \
  --bucket your-bucket-name \
  --input-prefix data/01_raw/ais_dk/raw/ \
  --output-prefix data/03_primary/cleaned_ais/ \
  --max-files 1
```

### 5. Interactive Examples

Run guided examples:

```bash
python scripts/run_s3_pipeline.py
```

## Structure

### Core Scripts (`scripts/`)
- **`s3_ais_processor.py`** - Main S3-based AIS data processor
- **`run_s3_pipeline.py`** - Interactive examples and guided usage
- **`run_tests.py`** - Test runner for all validation tests
- **`download_ais_data.py`** - Download AIS data from Danish Maritime Authority
- **`fetch_ais_data.sh`** - Shell wrapper for easy data downloading

### Testing & Validation (`tests/`)
- **`comprehensive_test.py`** - Full pipeline validation suite
- **`test_s3_processor.py`** - Basic system validation tests  
- **`inspect_data.py`** - Data quality inspection and validation

### Setup & Utilities
- **`setup.sh`** - Automated setup script  
- **`config.yaml`** - Configuration file
- **`requirements.txt`** - Python dependencies

## Configuration

Edit `config.yaml` to adjust processing parameters:

```yaml
processing:
  speed_thresh: 80.0      # Max speed in knots
  gap_hours: 6.0          # Time gap for new tracks  
  interpolate: false      # Enable interpolation
  interpolate_interval: 10 # Minutes between points

s3:
  bucket_name: "your-bucket-name"
  input_prefix: "data/01_raw/ais_dk/raw/"
  output_prefix: "data/03_primary/cleaned_ais/"
```

## Data Pipeline

```
DMA Download → S3 Storage → CSV Extraction → Data Cleaning → Speed Filtering 
      ↓              ↓             ↓
   download_ais_data.py  →  s3_ais_processor.py  →  Track Creation → Validation → S3 Parquet Output
```

### Processing Steps

1. **Data Download**: Downloads ZIP files from Danish Maritime Authority to S3
2. **ZIP Extraction**: Streams CSV files from S3 ZIP archives  
3. **Column Mapping**: Standardizes various AIS column formats
4. **Data Cleaning**: Removes invalid coordinates and duplicates
5. **Speed Filtering**: Filters impossible vessel speeds (configurable threshold)
6. **Track Creation**: Groups position reports into voyages based on time gaps
7. **State Management**: Maintains track continuity across processing runs
8. **Validation**: Quality checks and statistics generation
9. **Output**: Compressed Parquet files stored in S3

## Performance

- **Processing Speed**: ~500MB ZIP file in <40 seconds
- **Memory Usage**: Constant ~1-2GB regardless of input size
- **Compression**: 5x reduction in storage (ZIP → Parquet)
- **Track Quality**: <3% single-point tracks (normal for stationary vessels)

## Advanced Usage

### Reset Track State

Start fresh tracking (loses continuity):

```bash
python scripts/s3_ais_processor.py --reset-state ...
```

### Custom Parameters

```bash
python scripts/s3_ais_processor.py \
  --speed-thresh 60.0 \
  --gap-hours 4.0 \
  --interpolate \
  --interpolate-interval 15
```

### Data Inspection

Analyze specific vessels:

```bash
python scripts/inspect_data.py --mmsi 123456789
```

## Output Format

Processed data is stored as Parquet files with the following schema:

- **timestamp**: Position report timestamp
- **mmsi**: Maritime Mobile Service Identity  
- **lat/lon**: Coordinates (decimal degrees)
- **sog**: Speed over ground (knots)
- **cog**: Course over ground (degrees)
- **heading**: Vessel heading (degrees)
- **ship_type**: Vessel type
- **track_id**: Unique track identifier (MMSI_sequence)
- **date**: Processing date for partitioning

## AWS Requirements

### IAM Permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject", 
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*"
      ]
    }
  ]
}
```

## Troubleshooting

### Common Issues

1. **S3 Access Denied**: Check AWS credentials and IAM permissions
2. **Memory Errors**: Reduce processing chunk size in config
3. **Empty Output**: Verify input file format and date range
4. **Slow Processing**: Check network connectivity and AWS region

### Testing & Validation

Run all tests:
```bash
python scripts/run_tests.py --bucket your-bucket-name
```

Or run individual tests:

Basic system validation:
```bash
python tests/test_s3_processor.py
```

Comprehensive pipeline testing:
```bash
python tests/comprehensive_test.py --bucket your-bucket-name --max-days 2
```

Data quality inspection:
```bash
python tests/inspect_data.py --bucket your-bucket-name
```

## License

See LICENSE file for details.