from typing import Dict, Any
import mlflow
import logging

logger = logging.getLogger(__name__)


class MLflowMetricsLogger:
    """
    Stateless MLflow metrics logger
    Assumes an active MLflow run
    """

    def __init__(
        self,
        log_every_n_steps: int = 100,
        step_metric: str = "step",
    ):
        self.log_every_n_steps = log_every_n_steps
        self.step_metric = step_metric

    def _log(self, metrics: Dict[str, Any], step: int):
        mlflow.log_metrics(metrics, step=step)

    def on_step(self, step: int, metrics: Dict[str, Any]):
        if step % self.log_every_n_steps != 0:
            return

        namespaced = {
            f"train/{k}": v for k, v in metrics.items()
        }
        self._log(namespaced, step)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        namespaced = {
            f"val/{k}": v for k, v in metrics.items()
        }
        self._log(namespaced, epoch)

class EarlyStopping:
    """
    Early stopping based on validation loss
    Shared for bi-encoder and cross-encoder
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop
        
        Returns:
            True if should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            logger.info(f"✓ New best validation loss: {val_loss:.4f}")
        else:
            self.counter += 1
            logger.info(f"⚠ No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("✗ Early stopping triggered!")

        return self.should_stop
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False