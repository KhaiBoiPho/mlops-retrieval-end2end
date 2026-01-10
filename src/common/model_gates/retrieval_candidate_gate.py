import json
from pathlib import Path
from typing import Tuple, Dict


class RetrievalModelGate:
    """
    Gate logic for Bi-Encoder + Cross-Encoder candidate evaluation.
    DAG should NEVER contain metric logic.
    """

    def __init__(
        self,
        bi_metrics_path: str,
        cross_metrics_path: str,
    ):
        self.bi_metrics_path = Path(bi_metrics_path)
        self.cross_metrics_path = Path(cross_metrics_path)

    def load_metrics(self) -> Tuple[Dict, Dict]:
        with open(self.bi_metrics_path) as f:
            bi_metrics = json.load(f)

        with open(self.cross_metrics_path) as f:
            cross_metrics = json.load(f)

        return bi_metrics, cross_metrics

    def is_bi_encoder_valid(self, bi_metrics: Dict) -> bool:
        return (
            bi_metrics["mrr"] >= 0.35
            and bi_metrics["recall@10"] >= 0.60
        )

    def is_cross_encoder_valid(self, cross_metrics: Dict) -> bool:
        # Absolute rule: cross-encoder must learn positives
        if cross_metrics["true_positives"] == 0:
            return False

        return (
            cross_metrics["auc_roc"] >= 0.65
            and cross_metrics["recall"] >= 0.05
        )

    def evaluate(self) -> Tuple[bool, Dict]:
        bi_metrics, cross_metrics = self.load_metrics()

        bi_ok = self.is_bi_encoder_valid(bi_metrics)
        cross_ok = self.is_cross_encoder_valid(cross_metrics)

        details = {
            "bi_encoder_pass": bi_ok,
            "cross_encoder_pass": cross_ok,
            "bi_mrr": bi_metrics["mrr"],
            "bi_recall@10": bi_metrics["recall@10"],
            "cross_auc": cross_metrics["auc_roc"],
            "cross_recall": cross_metrics["recall"],
            "cross_tp": cross_metrics["true_positives"],
        }

        return bi_ok and cross_ok, details