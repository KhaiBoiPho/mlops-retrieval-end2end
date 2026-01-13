# scripts/download_checkpoints_from_s3.py

"""
Script to download all checkpoints from S3 for a specific run
Usage: python scripts/download_checkpoints_from_s3.py --run-id abc123
"""

import argparse
from pathlib import Path
from tqdm import tqdm
from src.common.s3_utils import S3Client
from src.common.logging_config import get_logger

logger = get_logger(__name__)

def download_run_artifacts(
    run_id: str,
    s3_bucket: str,
    s3_prefix: str,
    local_dir: Path
):
    """Download all artifacts for a specific run from S3"""
    
    s3_client = S3Client()
    
    # List all files in S3 for this run
    s3_run_prefix = f"{s3_prefix}/{run_id}/"
    
    logger.info(f"📥 Downloading from s3://{s3_bucket}/{s3_run_prefix}")
    logger.info(f"📂 Saving to: {local_dir}")
    
    # Create local directory
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # List all S3 objects
    import boto3
    s3 = boto3.client('s3')
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_run_prefix)
    
    all_objects = []
    for page in pages:
        if 'Contents' in page:
            all_objects.extend(page['Contents'])
    
    logger.info(f"📦 Found {len(all_objects)} files to download")
    
    # Download all files
    downloaded = 0
    failed = 0
    
    with tqdm(total=len(all_objects), desc="Downloading") as pbar:
        for obj in all_objects:
            s3_key = obj['Key']
            
            # Calculate local path
            relative_path = s3_key.replace(s3_run_prefix, '')
            local_path = local_dir / relative_path
            
            # Create parent directory
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                s3_client.download_file(
                    s3_bucket=s3_bucket,
                    s3_key=s3_key,
                    local_path=local_path,
                    show_progress=False
                )
                downloaded += 1
            except Exception as e:
                logger.error(f"❌ Failed to download {relative_path}: {e}")
                failed += 1
            
            pbar.update(1)
            pbar.set_postfix({'downloaded': downloaded, 'failed': failed})
    
    logger.info(f"✓ Download complete: {downloaded} succeeded, {failed} failed")

def main():
    parser = argparse.ArgumentParser(description="Download training artifacts from S3")
    parser.add_argument('--run-id', required=True, help='MLflow run ID')
    parser.add_argument('--bucket', default='legal-retrieval-models', help='S3 bucket')
    parser.add_argument('--prefix', default='models/bi-encoder', help='S3 prefix')
    parser.add_argument('--output', default='./downloaded_artifacts', help='Local output directory')
    
    args = parser.parse_args()
    
    download_run_artifacts(
        run_id=args.run_id,
        s3_bucket=args.bucket,
        s3_prefix=args.prefix,
        local_dir=Path(args.output) / args.run_id
    )

if __name__ == "__main__":
    main()