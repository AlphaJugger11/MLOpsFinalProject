# Federated Learning Dashboard

A Streamlit-based dashboard for visualizing and interacting with the Federated Learning MLOps pipeline.

## Features

- 🔮 **Real-time Predictions**: Make predictions on human activity data
- 📈 **Training Metrics**: Visualize federated learning training progress
- 🎯 **Model Performance**: View confusion matrix and per-class metrics
- 📊 **Interactive Visualizations**: Plotly-based charts and graphs
- ☁️ **S3 Integration**: Load models directly from AWS S3
- 🎨 **Clean UI**: Professional and intuitive interface

## Installation

```bash
cd dashboard
pip install -r requirements.txt
```

## Running the Dashboard

### Option 1: Local Mode (with local model)

```bash
streamlit run app.py
```

### Option 2: S3 Mode (with AWS credentials)

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1
export S3_BUCKET=mlopsprojbucket

# Run dashboard
streamlit run app.py
```

### Option 3: Docker

```bash
# Build dashboard image
docker build -t fl-dashboard -f Dockerfile .

# Run dashboard
docker run -p 8501:8501 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e S3_BUCKET=mlopsprojbucket \
  fl-dashboard
```

## Usage

1. **Load Model**:

   - Choose "Local" or "S3" from the sidebar
   - Click "Load Model" button
   - Wait for confirmation message

2. **Make Predictions**:

   - Go to "Predictions" tab
   - Choose input method:
     - Upload CSV file
     - Use sample data
     - Generate random sample
   - Click "Predict" button
   - View results and confidence scores

3. **View Training Metrics**:

   - Go to "Training Metrics" tab
   - See training progress over rounds
   - View client contributions

4. **Analyze Performance**:
   - Go to "Model Performance" tab
   - View confusion matrix
   - Check per-class metrics

## Dashboard Sections

### 🔮 Predictions Tab

- Upload test data or use samples
- Get real-time predictions
- View confidence scores for all activities
- See prediction accuracy (if labels available)

### 📈 Training Metrics Tab

- Training progress visualization
- Accuracy and loss curves
- Client participation statistics
- Round-by-round metrics

### 🎯 Model Performance Tab

- Overall model metrics (accuracy, precision, recall, F1)
- Confusion matrix heatmap
- Per-class performance breakdown
- Support statistics

### ℹ️ About Tab

- Project overview
- Architecture description
- Dataset information
- Technology stack
- Quick start guide

## Configuration

Environment variables:

- `S3_BUCKET`: S3 bucket name (default: mlopsprojbucket)
- `AWS_REGION`: AWS region (default: us-east-1)
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key

## Model Requirements

The dashboard expects PyTorch models with:

- Input dimension: 561 features
- Output dimension: 6 classes
- Model architecture: GlobalNet (from shared/model.py)

## Data Format

Input CSV should have:

- 561 feature columns (sensor readings)
- Optional: 'subject' column
- Optional: 'activity' column (for validation)

## Troubleshooting

### Model Not Loading

- Check model path is correct
- Verify AWS credentials for S3
- Ensure model file exists

### Prediction Errors

- Verify input has exactly 561 features
- Check data types (should be numeric)
- Ensure no NaN values

### S3 Connection Issues

- Verify AWS credentials are set
- Check S3 bucket name and region
- Ensure IAM permissions for S3 read access

## Screenshots

The dashboard includes:

- Clean, professional interface
- Interactive Plotly charts
- Real-time predictions
- Comprehensive metrics visualization

## Port

Default port: 8501

Access at: http://localhost:8501

## Notes

- The dashboard uses mock data for training history and confusion matrix
- Replace with actual MLflow data for production use
- Confidence threshold for warnings: 60%
- Supports both local and S3 model loading
