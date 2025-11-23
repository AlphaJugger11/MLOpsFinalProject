#!/bin/bash
# Script to run the Streamlit dashboard

set -e

echo "🚀 Starting Federated Learning Dashboard..."

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Set default environment variables if not set
export S3_BUCKET=${S3_BUCKET:-"mlopsprojbucket"}
export AWS_REGION=${AWS_REGION:-"us-east-1"}

echo "📊 Configuration:"
echo "  S3 Bucket: $S3_BUCKET"
echo "  AWS Region: $AWS_REGION"

# Check if AWS credentials are set
if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "⚠️  Warning: AWS_ACCESS_KEY_ID not set. S3 features may not work."
fi

# Run Streamlit
echo "🌐 Dashboard will be available at: http://localhost:8501"
streamlit run app.py
