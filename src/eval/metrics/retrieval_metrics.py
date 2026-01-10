import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """
    Compute retrieval metrics for bi-encoder evaluation
    """
    
    @staticmethod
    def recall_at_k(
        predictions: List[List[int]],
        ground_truths: List[List[int]],
        k_values: List[int]
    ) -> Dict[str, float]:
        """
        Compute Recall@K
        
        Args:
            predictions: List of ranked document IDs for each query
            ground_truths: List of relevant document IDs for each query
            k_values: List of K values to compute recall
            
        Returns:
            Dict of Recall@K scores
        """
        recalls = {}
        
        for k in k_values:
            correct = 0
            total = len(predictions)
            
            for pred, truth in zip(predictions, ground_truths):
                # Get top-k predictions
                top_k = pred[:k]
                
                # Check if any relevant doc in top-k
                if any(doc_id in truth for doc_id in top_k):
                    correct += 1
            
            recalls[f'recall@{k}'] = correct / total if total > 0 else 0.0
        
        return recalls
    
    @staticmethod
    def precision_at_k(
        predictions: List[List[int]],
        ground_truths: List[List[int]],
        k_values: List[int]
    ) -> Dict[str, float]:
        """
        Compute Precision@K
        
        Args:
            predictions: List of ranked document IDs
            ground_truths: List of relevant document IDs
            k_values: List of K values
            
        Returns:
            Dict of Precision@K scores
        """
        precisions = {}
        
        for k in k_values:
            total_precision = 0.0
            total = len(predictions)
            
            for pred, truth in zip(predictions, ground_truths):
                top_k = pred[:k]
                
                # Count relevant docs in top-k
                relevant_in_k = sum(1 for doc_id in top_k if doc_id in truth)
                total_precision += relevant_in_k / k
            
            precisions[f'precision@{k}'] = total_precision / total if total > 0 else 0.0
        
        return precisions
    
    @staticmethod
    def mean_reciprocal_rank(
        predictions: List[List[int]],
        ground_truths: List[List[int]]
    ) -> float:
        """
        Compute MRR (Mean Reciprocal Rank)
        
        Returns:
            MRR score
        """
        reciprocal_ranks = []
        
        for pred, truth in zip(predictions, ground_truths):
            # Find rank of first relevant document
            for rank, doc_id in enumerate(pred, start=1):
                if doc_id in truth:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                # No relevant doc found
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    @staticmethod
    def mean_average_precision(
        predictions: List[List[int]],
        ground_truths: List[List[int]]
    ) -> float:
        """
        Compute MAP (Mean Average Precision)
        
        Returns:
            MAP score
        """
        average_precisions = []
        
        for pred, truth in zip(predictions, ground_truths):
            if not truth:
                continue
            
            precision_sum = 0.0
            num_relevant = 0
            
            for rank, doc_id in enumerate(pred, start=1):
                if doc_id in truth:
                    num_relevant += 1
                    precision_at_rank = num_relevant / rank
                    precision_sum += precision_at_rank
            
            if num_relevant > 0:
                avg_precision = precision_sum / len(truth)
                average_precisions.append(avg_precision)
        
        return np.mean(average_precisions) if average_precisions else 0.0
    
    @staticmethod
    def ndcg_at_k(
        predictions: List[List[int]],
        ground_truths: List[List[int]],
        k_values: List[int]
    ) -> Dict[str, float]:
        """
        Compute NDCG@K (Normalized Discounted Cumulative Gain)
        
        Returns:
            Dict of NDCG@K scores
        """
        ndcg_scores = {}
        
        for k in k_values:
            ndcg_list = []
            
            for pred, truth in zip(predictions, ground_truths):
                top_k = pred[:k]
                
                # DCG - Discounted Cumulative Gain
                dcg = 0.0
                for rank, doc_id in enumerate(top_k, start=1):
                    if doc_id in truth:
                        # Relevance = 1 for relevant docs
                        dcg += 1.0 / np.log2(rank + 1)
                
                # IDCG - Ideal DCG (all relevant docs at top)
                idcg = 0.0
                num_relevant = min(len(truth), k)
                for rank in range(1, num_relevant + 1):
                    idcg += 1.0 / np.log2(rank + 1)
                
                # NDCG
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_list.append(ndcg)
            
            ndcg_scores[f'ndcg@{k}'] = np.mean(ndcg_list) if ndcg_list else 0.0
        
        return ndcg_scores
    
    @staticmethod
    def hit_rate_at_k(
        predictions: List[List[int]],
        ground_truths: List[List[int]],
        k_values: List[int]
    ) -> Dict[str, float]:
        """
        Compute Hit Rate@K (Success Rate@K)
        Same as Recall@K for single relevant document
        
        Returns:
            Dict of Hit Rate@K scores
        """
        hit_rates = {}
        
        for k in k_values:
            hits = 0
            total = len(predictions)
            
            for pred, truth in zip(predictions, ground_truths):
                top_k = pred[:k]
                if any(doc_id in truth for doc_id in top_k):
                    hits += 1
            
            hit_rates[f'hit_rate@{k}'] = hits / total if total > 0 else 0.0
        
        return hit_rates
    
    @staticmethod
    def compute_all_metrics(
        predictions: List[List[int]],
        ground_truths: List[List[int]],
        recall_k: List[int] = [1, 3, 5, 10, 20, 50, 100],
        precision_k: List[int] = [1, 3, 5, 10, 20],
        ndcg_k: List[int] = [1, 3, 5, 10, 20]
    ) -> Dict[str, float]:
        """
        Compute all metrics at once
        
        Returns:
            Dict with all metric scores
        """
        metrics = {}
        
        # Recall@K
        metrics.update(RetrievalMetrics.recall_at_k(predictions, ground_truths, recall_k))
        
        # Precision@K
        metrics.update(RetrievalMetrics.precision_at_k(predictions, ground_truths, precision_k))
        
        # NDCG@K
        metrics.update(RetrievalMetrics.ndcg_at_k(predictions, ground_truths, ndcg_k))
        
        # Hit Rate@K
        metrics.update(RetrievalMetrics.hit_rate_at_k(predictions, ground_truths, recall_k))
        
        # MRR
        metrics['mrr'] = RetrievalMetrics.mean_reciprocal_rank(predictions, ground_truths)
        
        # MAP
        metrics['map'] = RetrievalMetrics.mean_average_precision(predictions, ground_truths)
        
        return metrics