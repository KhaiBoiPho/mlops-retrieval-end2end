# src/data/processors.py
import pandas as pd
import ast
from pathlib import Path
from tqdm import tqdm
from pyvi import ViTokenizer
from joblib import Parallel, delayed
import multiprocessing as mp
from sklearn.model_selection import train_test_split

from src.common.logging_config import get_logger

logger = get_logger(__name__)


# ==================== CSV PREPROCESSOR ====================
class CSVPreprocessor:
    """Basic CSV preprocessing - parse lists, clean data"""
    
    def __init__(self, raw_data_dir: Path, processed_data_dir: Path):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.n_jobs = mp.cpu_count()
    
    def preprocess_train(
        self, 
        input_file: str = "train.csv", 
        output_file: str = "train_cleaned.csv"
    ) -> Path:
        """Preprocess train.csv"""
        logger.info(f"Preprocessing {input_file}...")
        
        df = pd.read_csv(self.raw_data_dir / input_file)
        
        # Parse context column (parallel)
        logger.info("Parsing context...")
        df['context'] = Parallel(n_jobs=self.n_jobs, batch_size=100)(
            delayed(self._parse_list_column)(x) for x in tqdm(df['context'])
        )
        
        # Parse cid column (parallel)
        logger.info("Parsing cid...")
        df['cid'] = Parallel(n_jobs=self.n_jobs, batch_size=100)(
            delayed(self._parse_cid_column)(x) for x in tqdm(df['cid'])
        )
        
        # Remove rows with invalid cid
        df = df.dropna(subset=['cid'])
        df['cid'] = df['cid'].astype(int)
        
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_data_dir / output_file
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved to {output_path} ({len(df)} samples)")
        
        return output_path
    
    def preprocess_corpus(
        self, 
        input_file: str = "corpus.csv", 
        output_file: str = "corpus_cleaned.csv"
    ) -> Path:
        """Preprocess corpus.csv"""
        logger.info(f"Preprocessing {input_file}...")
        
        df = pd.read_csv(self.raw_data_dir / input_file)
        
        # Basic validation
        df = df.dropna(subset=['text', 'cid'])
        df['cid'] = df['cid'].astype(int)
        
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_data_dir / output_file
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved to {output_path} ({len(df)} documents)")
        
        return output_path
    
    @staticmethod
    def _parse_list_column(x):
        try:
            lst = ast.literal_eval(x)
            return lst if isinstance(lst, list) else [str(lst)]
        except Exception:
            return [str(x)]
    
    @staticmethod
    def _parse_cid_column(x):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, list) and len(val) > 0:
                return val[0]
            return val
        except Exception:
            return int(x) if str(x).isdigit() else None


# ==================== VIETNAMESE TOKENIZER ====================
class VietnameseTokenizer:
    """Vietnamese word segmentation using PyVi"""
    
    def __init__(self, processed_data_dir: Path, use_data_dir: Path):
        self.processed_data_dir = processed_data_dir
        self.use_data_dir = use_data_dir
        self.n_jobs = mp.cpu_count()
    
    def tokenize_train(
        self, 
        input_file: str = "train_cleaned.csv", 
        output_file: str = "train_tokenized.csv"
    ) -> Path:
        """Tokenize train data"""
        logger.info(f"Tokenizing {input_file}...")
        
        df = pd.read_csv(self.processed_data_dir / input_file)
        
        # Tokenize questions
        logger.info("Tokenizing questions...")
        df['question'] = Parallel(n_jobs=self.n_jobs, batch_size=100)(
            delayed(self._tokenize_text)(q) for q in tqdm(df['question'])
        )
        
        # Tokenize contexts
        logger.info("Tokenizing contexts...")
        df['context'] = Parallel(n_jobs=self.n_jobs, batch_size=100)(
            delayed(self._tokenize_context_list)(x) for x in tqdm(df['context'])
        )
        
        self.use_data_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.use_data_dir / output_file
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved to {output_path}")
        
        return output_path
    
    def tokenize_corpus(
        self, 
        input_file: str = "corpus_cleaned.csv", 
        output_file: str = "corpus_tokenized.csv"
    ) -> Path:
        """Tokenize corpus data"""
        logger.info(f"Tokenizing {input_file}...")
        
        df = pd.read_csv(self.processed_data_dir / input_file)
        
        # Tokenize text
        logger.info("Tokenizing corpus text...")
        df['text'] = Parallel(n_jobs=self.n_jobs, batch_size=100)(
            delayed(self._tokenize_text)(x) for x in tqdm(df['text'])
        )
        
        self.use_data_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.use_data_dir / output_file
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved to {output_path} ({len(df)} docs)")
        
        return output_path
    
    def split_train_data(
        self,
        input_file: str = "train_tokenized.csv",
        train_size: float = 0.7,
        val_size: float = 0.2,
        test_size: float = 0.1,
        random_state: int = 42
    ) -> tuple[Path, Path, Path]:
        """Split train data into train/val/test"""
        logger.info(f"Splitting data: {train_size*100:.0f}/{val_size*100:.0f}/{test_size*100:.0f}%")
        
        df = pd.read_csv(self.use_data_dir / input_file)
        
        # First split: train vs temp (val+test)
        temp_size = val_size + test_size
        train_df, temp_df = train_test_split(df, test_size=temp_size, random_state=random_state)
        
        # Second split: val vs test
        test_ratio = test_size / temp_size
        val_df, test_df = train_test_split(temp_df, test_size=test_ratio, random_state=random_state)
        
        # Save splits
        train_path = self.use_data_dir / "train_split.csv"
        val_path = self.use_data_dir / "val_split.csv"
        test_path = self.use_data_dir / "test_split.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"✓ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_path, val_path, test_path
    
    @staticmethod
    def _tokenize_text(text):
        if not isinstance(text, str):
            return ""
        return ViTokenizer.tokenize(text)
    
    def _tokenize_context_list(self, context):
        try:
            if isinstance(context, str):
                context = ast.literal_eval(context)
            if isinstance(context, list):
                return [self._tokenize_text(item) for item in context]
            return [self._tokenize_text(str(context))]
        except Exception:
            return [self._tokenize_text(str(context))]


# ==================== DATA VALIDATOR ====================
class DataValidator:
    """Validate data quality"""
    
    @staticmethod
    def validate_train_data(file_path: Path) -> bool:
        """Validate train data"""
        logger.info(f"Validating {file_path.name}...")
        
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_cols = ['question', 'context', 'cid', 'qid']
        missing = set(required_cols) - set(df.columns)
        if missing:
            logger.error(f"Missing columns: {missing}")
            return False
        
        # Check for nulls
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values:\n{null_counts[null_counts > 0]}")
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(df['cid']):
            logger.error("cid must be numeric")
            return False
        
        logger.info(f"✓ Valid: {len(df)} rows")
        return True
    
    @staticmethod
    def validate_corpus_data(file_path: Path) -> bool:
        """Validate corpus data"""
        logger.info(f"Validating {file_path.name}...")
        
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_cols = ['text', 'cid']
        missing = set(required_cols) - set(df.columns)
        if missing:
            logger.error(f"Missing columns: {missing}")
            return False
        
        # Check for nulls
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            logger.error(f"Null values:\n{null_counts[null_counts > 0]}")
            return False
        
        # Check duplicates
        dup_cids = df['cid'].duplicated().sum()
        if dup_cids > 0:
            logger.warning(f"Found {dup_cids} duplicate cids")
        
        logger.info(f"✓ Valid: {len(df)} documents")
        return True