#!/bin/bash
# AIS Pipeline Setup Script

echo "Setting up AIS Data Processing Pipeline..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python3."
    exit 1
fi

echo "✅ Python3 found"

# Install pip if needed
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip..."
    curl -s https://bootstrap.pypa.io/get-pip.py | python3
fi

echo "✅ Pip available"

# Install dependencies
echo "Installing Python dependencies..."
python3 -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check AWS CLI (optional)
if command -v aws &> /dev/null; then
    echo "✅ AWS CLI found"
    echo "Make sure to configure AWS credentials:"
    echo "  aws configure"
else
    echo "⚠️  AWS CLI not found (optional)"
    echo "Install with: pip install awscli"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Configure AWS credentials: aws configure"
echo "2. Update config.yaml with your S3 bucket name"  
echo "3. Run basic test: python3 tests/test_s3_processor.py"
echo "4. Run full tests: python3 scripts/run_tests.py --bucket your-bucket-name"
echo "5. Process data: python3 scripts/run_s3_pipeline.py"