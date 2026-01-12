"""
Update S3 'latest' folder to point to newest trained model
Run after training completes
"""
import boto3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env
project_root = Path(__file__).parent.parent
load_dotenv(project_root / '.env')

s3 = boto3.client('s3')
BUCKET = os.getenv('S3_BUCKET')


def copy_folder_in_s3(source_prefix: str, dest_prefix: str):
    """Copy all files from source to dest in S3"""
    print(f"Copying from {source_prefix} to {dest_prefix}")
    
    # List source files
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=source_prefix)
    
    count = 0
    for page in pages:
        for obj in page.get('Contents', []):
            source_key = obj['Key']
            
            # Calculate destination key
            relative_path = source_key.replace(source_prefix, '')
            dest_key = dest_prefix + relative_path
            
            # Copy
            s3.copy_object(
                Bucket=BUCKET,
                CopySource={'Bucket': BUCKET, 'Key': source_key},
                Key=dest_key
            )
            count += 1
            print(f"  ✓ Copied: {relative_path}")
    
    print(f"✓ Total files copied: {count}")


def update_latest_pointer(model_name: str, run_id: str):
    """Update 'latest' folder to point to specific run_id"""
    print(f"\n{'='*60}")
    print(f"Updating {model_name} latest → {run_id}")
    print(f"{'='*60}")
    
    source_prefix = f"models/{model_name}/{run_id}/best_model/"
    dest_prefix = f"models/{model_name}/latest/"
    
    # Delete old 'latest' folder
    print("Cleaning old latest folder...")
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=dest_prefix)
    
    for page in pages:
        for obj in page.get('Contents', []):
            s3.delete_object(Bucket=BUCKET, Key=obj['Key'])
    
    # Copy new version
    copy_folder_in_s3(source_prefix, dest_prefix)
    
    print(f"✅ {model_name} latest updated to {run_id}")
    print(f"   S3 path: s3://{BUCKET}/{dest_prefix}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python update_s3_latest.py <model_name> <run_id>")
        print("Example: python update_s3_latest.py bi-encoder abc123")
        sys.exit(1)
    
    model_name = sys.argv[1]  # 'bi-encoder' or 'cross-encoder'
    run_id = sys.argv[2]       # MLflow run ID
    
    update_latest_pointer(model_name, run_id)
