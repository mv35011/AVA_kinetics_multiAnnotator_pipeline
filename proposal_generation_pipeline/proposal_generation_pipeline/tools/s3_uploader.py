import os
import logging
from pathlib import Path
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class S3Uploader:
    """A client to handle uploading files and directories to an AWS S3 bucket."""

    def __init__(self, bucket_name: str, region_name: str = 'us-east-1'):
        """
        Initializes the S3Uploader.

        Args:
            bucket_name (str): The name of the S3 bucket.
            region_name (str): The AWS region of the bucket.
        """
        self.bucket_name = bucket_name
        try:
            self.s3_client = boto3.client('s3', region_name=region_name)
            # A simple check to see if credentials are valid
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"✓ Successfully connected to S3 bucket '{self.bucket_name}'.")
        except (NoCredentialsError, PartialCredentialsError):
            logger.error("✗ AWS credentials not found. Please configure them (e.g., via 'aws configure').")
            raise
        except Exception as e:
            logger.error(f"✗ Failed to connect to S3 bucket '{self.bucket_name}': {e}")
            raise

    def upload_directory(self, local_directory: str, s3_prefix: str):
        """
        Uploads the contents of a local directory to a specified prefix in the S3 bucket.

        Args:
            local_directory (str): The path to the local directory to upload.
            s3_prefix (str): The folder/prefix in the S3 bucket where files will be placed.
        """
        local_path = Path(local_directory)
        if not local_path.is_dir():
            logger.error(f"Local directory not found: {local_directory}")
            return

        files_to_upload = [f for f in local_path.rglob('*') if f.is_file()]
        if not files_to_upload:
            logger.warning(f"No files found to upload in {local_directory}.")
            return

        logger.info(f"Starting upload of {len(files_to_upload)} files from '{local_path.name}' to 's3://{self.bucket_name}/{s3_prefix}'...")

        for file_path in tqdm(files_to_upload, desc=f"  -> Uploading {local_path.name}"):
            # Create a relative path to maintain directory structure in S3
            relative_path = file_path.relative_to(local_path)
            s3_key = os.path.join(s3_prefix, str(relative_path)).replace("\\", "/") # S3 uses forward slashes

            try:
                self.s3_client.upload_file(str(file_path), self.bucket_name, s3_key)
            except Exception as e:
                logger.error(f"✗ Failed to upload {file_path} to {s3_key}: {e}")

        logger.info(f"✅ Upload complete for directory '{local_path.name}'.")