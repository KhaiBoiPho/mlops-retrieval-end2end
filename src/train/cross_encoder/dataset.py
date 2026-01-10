import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Dict

from src.common.logging_config import get_logger

logger = get_logger(__name__)


class CrossEncoderDataset(Dataset):
    """
    Dataset for cross-encoder training
    
    Loads from train_cross_encoder.csv with columns:
    - query: question text
    - positive_cid: CID of positive document
    - negative_cid: CID of negative document (or None)
    - label: 1 for positive, 0 for negative
    """
    
    def __init__(
        self,
        data_path: str,
        corpus_path: str,
        tokenizer,
        max_length: int = 512,
        max_samples: int = None
    ):
        self.data_path = data_path
        self.corpus_path = corpus_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.data = self._load_data()
        self.corpus = self._load_corpus()

        # Sample data if specified (MOVE THIS AFTER LOADING)
        if max_samples is not None and max_samples < len(self.data):
            self.data = self.data.sample(n=max_samples, random_state=42).reset_index(drop=True)
            logger.info(f"📊 Sampled {max_samples} from original dataset")
        
        logger.info("CrossEncoderDataset initialized")
        logger.info(f"  Samples: {len(self.data)}")
        logger.info(f"  Positive: {(self.data['label'] == 1).sum()}")
        logger.info(f"  Negative: {(self.data['label'] == 0).sum()}")
    
    def _load_data(self) -> pd.DataFrame:
        """Load training data"""
        df = pd.read_csv(self.data_path)
        
        # Validate columns
        required_cols = ['query', 'positive_cid', 'label']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        return df
    
    def _load_corpus(self) -> Dict[int, str]:
        """Load corpus and create CID -> text mapping"""
        df = pd.read_csv(self.corpus_path)
        
        # Create mapping
        corpus_dict = {}
        for _, row in df.iterrows():
            corpus_dict[int(row['cid'])] = str(row['text'])
        
        logger.info(f"  Corpus documents: {len(corpus_dict)}")
        return corpus_dict
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single training example
        
        Returns:
            dict with keys: input_ids, attention_mask, token_type_ids, label
        """
        row = self.data.iloc[idx]
        
        query = str(row['query'])
        label = int(row['label'])
        
        # Get document text
        if label == 1:
            # Positive pair
            doc_cid = int(row['positive_cid'])
        else:
            # Negative pair
            doc_cid = int(row['negative_cid']) if pd.notna(row.get('negative_cid')) else int(row['positive_cid'])
        
        document = self.corpus.get(doc_cid, "")
        
        # Tokenize query + document
        encoding = self.tokenizer(
            query,
            document,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)
        }


# def create_cross_encoder_dataloader(
#     data_path: str,
#     corpus_path: str,
#     tokenizer,
#     batch_size: int = 16,
#     max_length: int = 512,
#     shuffle: bool = True,
#     num_workers: int = 0
# ) -> DataLoader:
#     """
#     Create DataLoader for cross-encoder training
#     """
#     dataset = CrossEncoderDataset(
#         data_path=data_path,
#         corpus_path=corpus_path,
#         tokenizer=tokenizer,
#         max_length=max_length
#     )
    
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=True
#     )
    
#     return dataloader

def create_cross_encoder_dataloader(
    data_path: str,
    corpus_path: str,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    max_samples: int = None
) -> DataLoader:
    """
    Create DataLoader for cross-encoder training
    """
    dataset = CrossEncoderDataset(
        data_path=data_path,
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader