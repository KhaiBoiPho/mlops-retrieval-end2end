from pathlib import Path

from src.common.s3_utils import S3Client
from src.common.logging_config import get_logger
from src.entity.config_entity import DataIngestionConfig

logger = get_logger(__name__)


class DataIngestion:
    """Download data from S3"""

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.s3_client = S3Client(region_name=config.s3_region)

    def download_raw_data(self) -> tuple[Path, Path]:
        """
            Download raw corpus and train data from S3
            
            Returns:
                Tuple of (corpus_path, train_path)
        """
        logger.info("=" * 80)
        logger.info("DOWNLOADING RAW DATA FROM S3")
        logger.info("=" * 80)

        self.config.raw_data_dir.mkdir(parents=True, exist_ok=True)

        corpus_path = self.config.raw_data_dir / "corpus.csv"
        train_path = self.config.raw_data_dir / "train.csv"

        # Download corpus
        logger.info(f"Downloading corpus from s3://{self.config.s3_bucket}/{self.config.corpus_s3_key}")
        success = self.s3_client.download_file(
            s3_bucket=self.config.s3_bucket,
            s3_key=self.config.corpus_s3_key,
            local_path=corpus_path
        )
        if not success:
            raise Exception("Failed to download corpus from S3")
        
        # Download train
        logger.info(f"Downloading train from s3://{self.config.s3_bucket}/{self.config.train_s3_key}")
        success = self.s3_client.download_file(
            s3_bucket=self.config.s3_bucket,
            s3_key=self.config.train_s3_key,
            local_path=train_path
        )
        if not success:
            raise Exception("Failed to download train data from S3")
        
        logger.info("Raw data download completed")
        return corpus_path, train_path
    
    def download_processed_data(self) -> Path:
        """
            Download processed CSV files from S3 to processed_data directory
            
            Returns:
                Path to processed_data directory
        """
        logger.info("=" * 80)
        logger.info("DOWNLOADING PROCESSED DATA FROM S3")
        logger.info("=" * 80)
        
        self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        files_to_download = [
            (self.config.corpus_cleaned_s3_key, "corpus_cleaned.csv"),
            (self.config.train_cleaned_s3_key, "train_cleaned.csv")
        ]
        
        for s3_key, filename in files_to_download:
            local_path = self.config.processed_data_dir / filename
            logger.info(f"Downloading {filename}...")
            success = self.s3_client.download_file(
                bucket=self.config.s3_bucket,
                s3_key=s3_key,
                local_path=local_path
            )
            if not success:
                raise Exception(f"Failed to download {filename} from S3")
        
        logger.info("Processed data download completed")
        return self.config.processed_data_dir
    
    def download_use_data(self) -> Path:
        """
            Download tokenized and split CSV files from S3 to use_data directory
            
            Returns:
                Path to use_data directory
        """
        logger.info("=" * 80)
        logger.info("DOWNLOADING USE DATA FROM S3")
        logger.info("=" * 80)
        
        self.config.use_data_dir.mkdir(parents=True, exist_ok=True)
        
        files_to_download = [
            (self.config.corpus_tokenized_s3_key, "corpus_tokenized.csv"),
            (self.config.train_tokenized_s3_key, "train_tokenized.csv"),
            (self.config.train_split_s3_key, "train_split.csv"),
            (self.config.val_split_s3_key, "val_split.csv"),
            (self.config.test_split_s3_key, "test_split.csv")
        ]
        
        for s3_key, filename in files_to_download:
            local_path = self.config.use_data_dir / filename
            logger.info(f"Downloading {filename}...")
            success = self.s3_client.download_file(
                bucket=self.config.s3_bucket,
                s3_key=s3_key,
                local_path=local_path
            )
            if not success:
                logger.warning(f"Failed to download {filename}, skipping...")
                continue
        
        logger.info("Use data download completed")
        logger.info(f"\nFiles in {self.config.use_data_dir}:")
        for file in sorted(self.config.use_data_dir.glob("*.csv")):
            logger.info(f"  - {file.name}")
        
        return self.config.use_data_dir