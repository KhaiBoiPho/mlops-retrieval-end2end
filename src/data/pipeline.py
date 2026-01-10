# src/data/pipeline.py
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.configuration import ConfigurationManager
from src.data.data_ingestion import DataIngestion
from src.data.processors import CSVPreprocessor, VietnameseTokenizer, DataValidator
from src.common.logging_config import get_logger

logger = get_logger(__name__)


def download_raw():
    """Stage 1: Download raw data from S3"""
    logger.info("="*80)
    logger.info("STAGE 1: Download Raw Data")
    logger.info("="*80)
    
    config_manager = ConfigurationManager()
    config = config_manager.get_data_ingestion_config()
    
    ingestion = DataIngestion(config)
    ingestion.download_raw_data()
    
    logger.info("✓ Stage 1 completed\n")


def preprocess():
    """Stage 2: Preprocess raw data"""
    logger.info("="*80)
    logger.info("STAGE 2: Preprocess Data")
    logger.info("="*80)
    
    config_manager = ConfigurationManager()
    config = config_manager.get_data_preprocess_config()
    
    preprocessor = CSVPreprocessor(
        raw_data_dir=config.raw_data_dir,
        processed_data_dir=config.processed_data_dir
    )
    
    preprocessor.preprocess_train()
    preprocessor.preprocess_corpus()
    
    logger.info("✓ Stage 2 completed\n")


def tokenize():
    """Stage 3: Tokenize data"""
    logger.info("="*80)
    logger.info("STAGE 3: Tokenize Data")
    logger.info("="*80)
    
    config_manager = ConfigurationManager()
    config = config_manager.get_data_preprocess_config()
    
    tokenizer = VietnameseTokenizer(
        processed_data_dir=config.processed_data_dir,
        use_data_dir=config.use_data_dir
    )
    
    tokenizer.tokenize_train()
    tokenizer.tokenize_corpus()
    
    logger.info("✓ Stage 3 completed\n")


def split_data():
    """Stage 4: Split data"""
    logger.info("="*80)
    logger.info("STAGE 4: Split Data")
    logger.info("="*80)
    
    config_manager = ConfigurationManager()
    config = config_manager.get_data_preprocess_config()
    
    tokenizer = VietnameseTokenizer(
        processed_data_dir=config.processed_data_dir,
        use_data_dir=config.use_data_dir
    )
    
    tokenizer.split_train_data(
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        random_state=config.random_state
    )
    
    logger.info("✓ Stage 4 completed\n")


def validate():
    """Stage 5: Validate data"""
    logger.info("="*80)
    logger.info("STAGE 5: Validate Data")
    logger.info("="*80)
    
    config_manager = ConfigurationManager()
    config = config_manager.get_data_preprocess_config()
    
    validator = DataValidator()
    
    valid_train = validator.validate_train_data(config.use_data_dir / "train_split.csv")
    valid_val = validator.validate_train_data(config.use_data_dir / "val_split.csv")
    valid_test = validator.validate_train_data(config.use_data_dir / "test_split.csv")
    valid_corpus = validator.validate_corpus_data(config.use_data_dir / "corpus_tokenized.csv")
    
    # Save metrics
    Path("reports").mkdir(exist_ok=True)
    metrics = {
        "train_valid": valid_train,
        "val_valid": valid_val,
        "test_valid": valid_test,
        "corpus_valid": valid_corpus,
        "all_valid": all([valid_train, valid_val, valid_test, valid_corpus])
    }
    
    with open("reports/validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    if not metrics["all_valid"]:
        raise ValueError("Validation failed!")
    
    logger.info("✓ Stage 5 completed\n")


# Entry points for DVC
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <stage>")
        print("Stages: download_raw, preprocess, tokenize, split_data, validate")
        sys.exit(1)
    
    stage = sys.argv[1]
    
    try:
        if stage == "download_raw":
            download_raw()
        elif stage == "preprocess":
            preprocess()
        elif stage == "tokenize":
            tokenize()
        elif stage == "split_data":
            split_data()
        elif stage == "validate":
            validate()
        else:
            print(f"Unknown stage: {stage}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Stage '{stage}' failed: {e}")
        sys.exit(1)