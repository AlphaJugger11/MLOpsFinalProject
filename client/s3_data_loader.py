# client/s3_data_loader.py
import os
import time
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from pathlib import Path


class S3DownloadError(Exception):
    """Custom exception for S3 download failures"""
    pass


def download_client_data_from_s3(
    bucket_name: str,
    client_id: str,
    local_path: str = "/tmp/client_data.csv",
    max_retries: int = 3,
    base_delay: float = 1.0
) -> str:
    """
    Downloads client data partition from S3 with retry logic.
    
    Args:
        bucket_name: S3 bucket containing datasets
        client_id: Client identifier (e.g., "client_1", "1")
        local_path: Local path to save downloaded file
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        
    Returns:
        Path to downloaded file
        
    Raises:
        S3DownloadError: If download fails after all retries
    """
    # Normalize client_id to match S3 key format (client1, client2, client3)
    if client_id.startswith("client_"):
        # Remove underscore: client_1 -> client1
        client_id = client_id.replace("_", "")
    elif not client_id.startswith("client"):
        # Add prefix: 1 -> client1
        client_id = f"client{client_id}"
    
    # Files are at root level, not in datasets/ folder
    s3_key = f"{client_id}.csv"
    
    # Ensure local directory exists
    local_dir = Path(local_path).parent
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize S3 client
    try:
        s3_client = boto3.client('s3')
    except (NoCredentialsError, PartialCredentialsError) as e:
        raise S3DownloadError(f"AWS credentials not configured: {e}")
    
    # Retry loop with exponential backoff
    for attempt in range(max_retries):
        try:
            print(f"[s3_loader] Attempt {attempt + 1}/{max_retries}: Downloading {s3_key} from bucket {bucket_name}")
            
            # Download file from S3
            s3_client.download_file(bucket_name, s3_key, local_path)
            
            # Verify file was downloaded and has content
            if not os.path.exists(local_path):
                raise S3DownloadError(f"File not found after download: {local_path}")
            
            file_size = os.path.getsize(local_path)
            if file_size == 0:
                raise S3DownloadError(f"Downloaded file is empty: {local_path}")
            
            print(f"[s3_loader] Successfully downloaded {s3_key} ({file_size} bytes) to {local_path}")
            return local_path
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_msg = e.response.get('Error', {}).get('Message', str(e))
            
            if error_code == '404' or error_code == 'NoSuchKey':
                # File not found - no point in retrying
                raise S3DownloadError(
                    f"File not found in S3: s3://{bucket_name}/{s3_key}. "
                    f"Error: {error_msg}"
                )
            elif error_code == '403' or error_code == 'AccessDenied':
                # Permission error - no point in retrying
                raise S3DownloadError(
                    f"Access denied to S3 bucket: {bucket_name}. "
                    f"Check IAM permissions. Error: {error_msg}"
                )
            elif error_code == 'NoSuchBucket':
                # Bucket doesn't exist - no point in retrying
                raise S3DownloadError(
                    f"S3 bucket does not exist: {bucket_name}. Error: {error_msg}"
                )
            else:
                # Network or other transient error - retry
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"[s3_loader] Download failed with error {error_code}: {error_msg}. "
                          f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    raise S3DownloadError(
                        f"Failed to download after {max_retries} attempts. "
                        f"Last error ({error_code}): {error_msg}"
                    )
                    
        except Exception as e:
            # Unexpected error
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[s3_loader] Unexpected error: {e}. Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                raise S3DownloadError(
                    f"Failed to download after {max_retries} attempts. "
                    f"Last error: {e}"
                )
    
    # Should never reach here, but just in case
    raise S3DownloadError(f"Download failed after {max_retries} attempts")


def validate_data_integrity(file_path: str, expected_features: int = 561) -> bool:
    """
    Validates the integrity of downloaded CSV data.
    
    Args:
        file_path: Path to CSV file
        expected_features: Expected number of feature columns
        
    Returns:
        True if data is valid
        
    Raises:
        S3DownloadError: If data validation fails
    """
    try:
        import pandas as pd
        
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Check if file is empty
        if len(df) == 0:
            raise S3DownloadError(f"CSV file is empty: {file_path}")
        
        # Check for expected columns
        if 'subject' in df.columns and 'activity' in df.columns:
            feature_cols = [c for c in df.columns if c not in ('subject', 'activity')]
        else:
            # Assume last column is label
            feature_cols = df.columns[:-1].tolist()
        
        num_features = len(feature_cols)
        if num_features != expected_features:
            raise S3DownloadError(
                f"Invalid number of features: expected {expected_features}, got {num_features}"
            )
        
        # Check for NaN values
        if df[feature_cols].isnull().any().any():
            raise S3DownloadError(f"CSV contains NaN values in feature columns")
        
        print(f"[s3_loader] Data validation passed: {len(df)} samples, {num_features} features")
        return True
        
    except pd.errors.EmptyDataError:
        raise S3DownloadError(f"CSV file is empty or malformed: {file_path}")
    except Exception as e:
        raise S3DownloadError(f"Data validation failed: {e}")
