import torch
import pandas as pd
import ast
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.common.logging_config import get_logger
from src.common.utils import safe_int

logger = get_logger(__name__)


class BiEncoderDataset(Dataset):
    """
    PyTorch Dataset for bi-encoder contrastive learning
    Returns tokenized (query, positive_context) pairs
    """

    def __init__(
        self,
        data_path: Path,
        tokenizer_name: str,
        max_length: int
    ):
        self.data_path = data_path
        self.max_length = safe_int(max_length)

        # Load data
        logger.info(f"Loading data from {data_path}")
        self.df = pd.read_csv(data_path)[:1000]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Parse data
        self.queries = []
        self.positives = []
        self._parse_data()

        logger.info(f"Loaded {len(self)} samples")

    # def _parse_data(self):
    #     """Parse DataFrame into queries and positive contexts"""
    #     for _, row in self.df.iterrows():
    #         query = str(row["question"])
            
    #         # Parse context (may be string representation of list)
    #         context = row["context"]
    #         if isinstance(context, str) and context.startswith("["):
    #             try:
    #                 context_list = ast.literal_eval(context)
    #                 context = context_list[0] if context_list else ""
    #             except Exception:
    #                 pass
            
    #         self.queries.append(query)
    #         self.positives.append(str(context))

    def _parse_data(self):
        """
        Parse DataFrame into (query, positive) pairs with multi-positives
        Each query may have multiple positive contexts - we expand them all
        """
        original_count = len(self.df)
        skipped = 0
        
        for idx, row in self.df.iterrows():
            query = str(row["question"])
            context = row["context"]
            positives = []
            
            # context có thể là list dạng string
            if isinstance(context, str) and context.startswith("["):
                try:
                    positives = ast.literal_eval(context)
                except Exception as e:
                    logger.warning(f"Row {idx}: Failed to parse context list: {e}")
                    skipped += 1
                    continue
            else:
                positives = [context]
            
            # Take all positive contexts
            added = 0
            for pos in positives:
                if pos and str(pos).strip():
                    self.queries.append(query)
                    self.positives.append(str(pos).strip())
                    added += 1
            
            if added == 0:
                logger.warning(f"Row {idx}: No valid positive contexts found")
                skipped += 1
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped}/{original_count} rows due to parsing errors or empty contexts")
        
        logger.info(f"Expanded {original_count} queries to {len(self.queries)} training pairs")

    # def _parse_data(self, max_positives_per_query: int = None):
    #     """
    #     Parse DataFrame into (query, positive) pairs with multi-positives
        
    #     Args:
    #         max_positives_per_query: Limit number of positives per query (None = use all)
    #     """
    #     for idx, row in self.df.iterrows():
    #         query = str(row["question"])
    #         context = row["context"]
    #         positives = []
            
    #         if isinstance(context, str) and context.startswith("["):
    #             try:
    #                 positives = ast.literal_eval(context)
    #             except Exception as e:
    #                 logger.warning(f"Row {idx}: Failed to parse context list: {e}")
    #                 continue
    #         else:
    #             positives = [context]
            
    #         # Limit positives if specified
    #         if max_positives_per_query:
    #             positives = positives[:max_positives_per_query]
            
    #         # Bung hết positive contexts
    #         for pos in positives:
    #             if pos and str(pos).strip():
    #                 self.queries.append(query)
    #                 self.positives.append(str(pos).strip())

    def __len__(self) -> int:
        return len(self.queries)
    
    def __getitem__(self, index) -> Tuple[dict, dict]:
        """
        Returns:
            query_encoding: Tokenized query
            positive_encoding: Tokenized positive context
        """
        query = self.queries[index]
        positive = self.positives[index]

        # Tokenize query
        query_encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize positive context
        positive_encoding = self.tokenizer(
            positive,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Remove batch dimension
        query_encoding = {k: v.squeeze(0) for k, v in query_encoding.items()}
        positive_encoding = {k: v.squeeze(0) for k, v in positive_encoding.items()}
        
        return query_encoding, positive_encoding
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for DataLoader
        
        Args:
            batch: List of (query_encoding, positive_encoding) tuples
            
        Returns:
            queries: Dict of batched query tensors
            positives: Dict of batched positive tensors
        """
        queries = {}
        positives = {}

        # Stack all queries
        query_keys = batch[0][0].keys()
        for key in query_keys:
            queries[key] = torch.stack([item[0][key] for item in batch])

        # Stack all positives
        positive_keys = batch[0][1].keys()
        for key in positive_keys:
            positives[key] = torch.stack([item[1][key] for item in batch])

        return queries, positives
    
def create_dataloader(
    dataset: BiEncoderDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create DataLoader with custom collate function"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=BiEncoderDataset.collate_fn,
        pin_memory=True
    )
