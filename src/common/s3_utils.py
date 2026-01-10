import boto3
from src.common.logging_config import get_logger
from pathlib import Path
from typing import Optional, List
from botocore.exceptions import ClientError
from tqdm import tqdm

logger = get_logger(__name__)


class S3Client:
    """Handle S3 operations for data upload/download"""

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = None
    ):
        if aws_access_key_id and aws_secret_access_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
        else:
            self.s3_client = boto3.client('s3', region_name=region_name)

    def upload_file(
        self,
        local_path: Path,
        s3_bucket: str,
        s3_key: str,
        show_progress: bool = True
    ) -> bool:
        """Upload a file to S3 with progress bar"""
        try:
            if show_progress:
                file_size = local_path.stat().st_size
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {local_path.name}") as pbar:
                    self.s3_client.upload_file(
                        str(local_path),
                        s3_bucket,
                        s3_key,
                        Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
                    )
            else:
                self.s3_client.upload_file(str(local_path), s3_bucket, s3_key)

            logger.info(f"Uploaded to s3://{s3_bucket}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False
        
    def download_file(
        self,
        s3_bucket: str,
        s3_key: str,
        local_path: Path,
        show_progress: bool = True
    ):
        """Download a file from s3 with progress bar"""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if show_progress:
                response = self.s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
                file_size = response['ContentLength']

                with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {local_path.name}") as pbar:
                    self.s3_client.download_file(
                        s3_bucket,
                        s3_key,
                        str(local_path),
                        Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
                    )
            else:
                self.s3_client.download_file(s3_bucket, s3_key, str(local_path))

            logger.info(f"Downloaded to {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to download s3://{s3_bucket}/{s3_key}: {e}")
            return False
        
    def upload_directory(
        self, 
        local_dir: Path, 
        s3_bucket: str, 
        s3_prefix: str,
        file_patterns: List[str] = ["*.csv"]
    ) -> bool:
        """Upload multiple files from directory to S3"""
        files_to_upload = []
        for pattern in file_patterns:
            files_to_upload.extend(local_dir.glob(pattern))
        
        if not files_to_upload:
            logger.warning(f"No files matching {file_patterns} found in {local_dir}")
            return False
        
        logger.info(f"Found {len(files_to_upload)} files to upload")
        success = True
        
        for file_path in files_to_upload:
            s3_key = f"{s3_prefix}{file_path.name}"
            if not self.upload_file(file_path, s3_bucket, s3_key):
                success = False
        
        return success
    
    def download_directory(
        self,
        s3_bucket: str,
        s3_prefix: str,
        local_dir: Path,
        file_patterns: List[str] = ["*.csv"]
    ) -> bool:
        """Download multiple files from S3 to local directory"""
        try:
            # List all files with prefix
            response = self.s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
            
            if 'Contents' not in response:
                logger.warning(f"No files found in s3://{s3_bucket}/{s3_prefix}")
                return False
            
            # Filter by patterns
            files_to_download = []
            for obj in response['Contents']:
                s3_key = obj['Key']
                filename = Path(s3_key).name

                # Check if matches any pattern
                if any(Path(filename).match(pattern) for pattern in file_patterns):
                    files_to_download.append(s3_key)
            
            if not files_to_download:
                logger.warning(f"No files matching {file_patterns} found")
                return False
            
            logger.info(f"Found {len(files_to_download)} files to download")
            success = True
            
            for s3_key in files_to_download:
                filename = Path(s3_key).name
                local_path = local_dir / filename
                if not self.download_file(s3_bucket, s3_key, local_path):
                    success = False
            
            return success

        except ClientError as e:
            logger.error(f"Failed to download directory: {e}")
            return False
    
    def file_exists(self, s3_bucket: str, s3_key: str) -> bool:
        """Check if file exists in S3"""
        try:
            self.s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
            return True
        except ClientError:
            return False
        
    def list_files(self, bucket: str, prefix: str = "") -> List[str]:
        """List files in S3 bucket with given prefix"""
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except ClientError as e:
            logger.error(f"Failed to list files: {e}")
            return []