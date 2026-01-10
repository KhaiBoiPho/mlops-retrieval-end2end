import numpy as np
from typing import List, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.common.logging_config import get_logger

logger = get_logger(__name__)


class RerankMetrics:
    """
    Metrics for cross-encoder reranking evaluation
    Classification metrics for binary relevance prediction
    """
    
    @staticmethod
    def compute_all_metrics(
        predictions: List[float],
        labels: List[int],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute all reranking metrics
        
        Args:
            predictions: List of predicted scores (0-1)
            labels: List of ground truth labels (0 or 1)
            threshold: Decision threshold for binary classification
            
        Returns:
            Dictionary of metrics
        """
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Binary predictions based on threshold
        binary_preds = (predictions >= threshold).astype(int)
        
        # Accuracy
        accuracy = accuracy_score(labels, binary_preds)
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, binary_preds, average='binary', zero_division=0
        )
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(labels, predictions)
        except ValueError:
            # Handle case where only one class is present
            auc_roc = 0.0
            logger.warning("Cannot compute AUC-ROC (only one class present)")
        
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc_roc': float(auc_roc),
            
            # Confusion matrix components
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            
            # Additional useful metrics
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'sensitivity': float(recall),  # Same as recall
        }
        
        return metrics
    
    @staticmethod
    def print_classification_report(
        predictions: List[float],
        labels: List[int],
        threshold: float = 0.5
    ):
        """Print detailed classification report"""
        predictions = np.array(predictions)
        labels = np.array(labels)
        binary_preds = (predictions >= threshold).astype(int)
        
        logger.info("\n" + "="*80)
        logger.info("CLASSIFICATION REPORT")
        logger.info("="*80)
        
        report = classification_report(
            labels, 
            binary_preds,
            target_names=['Not Relevant', 'Relevant'],
            digits=4
        )
        
        logger.info("\n" + report)
    
    @staticmethod
    def compute_threshold_metrics(
        predictions: List[float],
        labels: List[int],
        thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
    ) -> Dict[float, Dict[str, float]]:
        """
        Compute metrics at different thresholds
        Useful for finding optimal threshold
        
        Args:
            predictions: Predicted scores
            labels: Ground truth labels
            thresholds: List of thresholds to evaluate
            
        Returns:
            Dictionary mapping threshold -> metrics
        """
        results = {}
        
        for threshold in thresholds:
            metrics = RerankMetrics.compute_all_metrics(
                predictions, labels, threshold
            )
            results[threshold] = metrics
        
        return results