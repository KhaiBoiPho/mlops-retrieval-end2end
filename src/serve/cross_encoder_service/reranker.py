import torch
from typing import List, Dict, Tuple
import time
# import numpy as np

from src.common.logging_config import get_logger
from src.entity.config_entity import CrossEncoderServeConfig

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Rerank documents using cross-encoder"""

    def __init__(
        self,
        model,
        tokenizer,
        config: CrossEncoderServeConfig
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        logger.info("CrossEncoderReranker initialized")
        logger.info(f"  Max length: {self.config.max_seq_length}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Device: {self.device}")

    @torch.no_grad()
    def score_pair(self, query: str, document: str) -> Tuple[float, float]:
        """
        Score a single query-document pair
        
        Returns:
            score: relevance score
            score_time: time in milliseconds
        """

        start_time = time.time()

        # Tokenize query-document pair
        inputs = self.tokenizer(
            query,
            document,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get score
        logits = self.model(**inputs)
        score = torch.sigmoid(logits).item()

        score_time = (time.time() - start_time) * 1000

        return score, score_time
    
    @torch.no_grad()
    def rerank(
        self,
        query: str,
        documents: List[Dict],  # ← Expects list of dicts with 'id' and 'text'
        top_n: int = 10
    ) -> Tuple[List[Dict], float]:
        """
        Rerank documents - ID is preserved through the pipeline
        
        Flow:
        1. Extract (id, text) pairs
        2. Score (query, text) pairs  ← Model doesn't see ID
        3. Sort by score
        4. Return results with original ID
        """
        start_time = time.time()
        
        if not documents:
            return [], 0.0
        
        # Step 1: Extract text for model (ID stored separately)
        pairs = [(query, doc['text']) for doc in documents]
        doc_ids = [doc['id'] for doc in documents]  # ← Store IDs
        
        # Step 2: Score pairs (model only sees text)
        all_scores = []
        for i in range(0, len(pairs), self.config.batch_size):
            batch_pairs = pairs[i:i + self.config.batch_size]
            batch_queries = [p[0] for p in batch_pairs]
            batch_docs = [p[1] for p in batch_pairs]
            
            inputs = self.tokenizer(
                batch_queries,
                batch_docs,  # ← Only text goes to model
                padding='max_length',
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            logits = self.model(**inputs)
            scores = torch.sigmoid(logits).squeeze(-1)
            all_scores.extend(scores.cpu().numpy())
        
        # Step 3: Combine scores with original IDs and documents
        scored_docs = [
            {
                'id': doc_ids[i],              # ← Original ID preserved
                'text': documents[i]['text'],
                'score': float(all_scores[i]),
                'metadata': documents[i].get('metadata')  # ← Metadata preserved
            }
            for i in range(len(documents))
        ]
        
        # Step 4: Sort and return
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        reranked_results = [
            {**doc, 'rank': idx + 1}  # ← Add rank, keep everything else
            for idx, doc in enumerate(scored_docs[:top_n])
        ]
        
        rerank_time = (time.time() - start_time) * 1000
        return reranked_results, rerank_time

    @torch.no_grad()
    def score_batch(
        self,
        query: str,
        documents: List[str]
    ) -> Tuple[List[float], float]:
        """
        Score multiple documents for a single query
        
        Returns:
            scores: list of relevance scores
            score_time: total time in milliseconds
        """
        start_time = time.time()

        if not documents:
            return [], 0.0
        
        all_scores = []

        # Process in batches
        for i in range(0, len(documents), self.config.batch_size):
            batch_docs = documents[i:i + self.config.batch_size]
            batch_queries = [query] * len(batch_docs)

            # Tokenize batch
            inputs = self.tokenizer(
                batch_queries,
                batch_docs,
                padding='max_length',
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get scores
            logits = self.model(**inputs)
            scores = torch.sigmoid(logits).squeeze(-1)

            all_scores.extend(scores.cpu().numpy())

        score_time = (time.time() - start_time) * 1000

        return [float(s) for s in all_scores], score_time
