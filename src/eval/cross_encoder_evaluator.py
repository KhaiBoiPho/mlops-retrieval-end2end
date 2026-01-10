import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import mlflow
import json
# import ast

from src.common.logging_config import get_logger
from src.models.cross_encoder.model import CrossEncoder
from src.entity.config_entity import CrossEncoderEvalConfig
from src.eval.metrics.rerank_metrics import RerankMetrics
from src.registry.mlflow_client import get_mlflow_client

logger = get_logger(__name__)


class CrossEncoderEvaluator:
    """
    Evaluate cross-encoder model on reranking task
    Computes: Accuracy, Precision, Recall, F1, AUC-ROC
    """
    
    def __init__(self, config: CrossEncoderEvalConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        self.mlflow_client = get_mlflow_client()
        
        self.model = None
        self.tokenizer = None
        self.corpus = None
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.config.batch_size}")
    
    def load_model(self):
        """Load trained cross-encoder model from MLflow"""
        logger.info("="*80)
        logger.info("LOADING MODEL FROM MLFLOW")
        logger.info("="*80)
        logger.info(f"Model name: {self.config.mlflow_model_name}")
        logger.info(f"Model stage: {self.config.mlflow_model_stage}")
        
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        
        # Build model URI
        if self.config.mlflow_run_id:
            model_uri = f"runs:/{self.config.mlflow_run_id}/best_model"
            logger.info(f"Loading from run ID: {self.config.mlflow_run_id}")
        elif self.config.mlflow_model_stage.lower() == "latest":
            # Get latest version from registry
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(
                f"name='{self.config.mlflow_model_name}'"
            )
            
            if not versions:
                raise ValueError(f"No versions found for model: {self.config.mlflow_model_name}")
            
            latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
            logger.info(f"Latest version found: {latest_version.version}")
            
            model_uri = f"models:/{self.config.mlflow_model_name}/{latest_version.version}"
            logger.info(f"Loading from registry: {model_uri}")
        else:
            model_uri = f"models:/{self.config.mlflow_model_name}/{self.config.mlflow_model_stage}"
            logger.info(f"Loading from registry: {model_uri}")
        
        # Download model artifacts from MLflow
        logger.info("Downloading model artifacts...")
        model_path = Path(mlflow.artifacts.download_artifacts(model_uri))
        
        # Try different possible locations for checkpoint
        possible_paths = [
            model_path / "pytorch_model.bin",
            model_path / "best_model.pt",
            model_path / "model.pt",
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if path.exists():
                checkpoint_path = path
                logger.info(f"Found checkpoint at: {checkpoint_path}")
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No model checkpoint found in {model_path}. "
                f"Tried: {[p.name for p in possible_paths]}"
            )
        
        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_config = checkpoint.get('config')
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model_config = checkpoint.get('config')
            state_dict = checkpoint['state_dict']
        else:
            # Direct state dict
            state_dict = checkpoint
            model_config = None
        
        if model_config is None:
            # Try to load from model_config.json
            config_path = model_path / "model_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    model_config_dict = json.load(f)
                
                class SimpleConfig:
                    def __init__(self, d):
                        for k, v in d.items():
                            setattr(self, k, v)
                
                model_config = SimpleConfig(model_config_dict)
            else:
                raise ValueError("Model config not found in checkpoint or model_config.json")
        
        # Initialize model
        self.model = CrossEncoder(
            model_name=model_config.model_name,
            num_labels=getattr(model_config, 'num_labels', 1)
        )
        
        # Load state dict
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model.encoder.config._name_or_path
        )
        
        logger.info(f"✓ Model loaded: {model_config.model_name}")
        logger.info(f"  Num labels: {self.model.num_labels}")
    
    def load_corpus(self):
        """Load corpus for getting document texts"""
        logger.info("="*80)
        logger.info("LOADING CORPUS")
        logger.info("="*80)
        logger.info(f"Path: {self.config.corpus_path}")
        
        if not self.config.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {self.config.corpus_path}")
        
        df = pd.read_csv(self.config.corpus_path)
        
        # Create CID -> text mapping
        self.corpus = {}
        for _, row in df.iterrows():
            self.corpus[int(row['cid'])] = str(row['text'])
        
        logger.info(f"✓ Loaded {len(self.corpus)} documents")
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data"""
        logger.info("="*80)
        logger.info("LOADING TEST DATA")
        logger.info("="*80)
        logger.info(f"Test path: {self.config.test_path}")
        
        if not self.config.test_path.exists():
            raise FileNotFoundError(f"Test data not found: {self.config.test_path}")
        
        df = pd.read_csv(self.config.test_path)
        
        # Validate columns
        required_cols = ['query', 'positive_cid', 'label']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        logger.info(f"✓ Loaded {len(df)} test samples")
        logger.info(f"  Positive: {(df['label'] == 1).sum()}")
        logger.info(f"  Negative: {(df['label'] == 0).sum()}")
        
        return df
    
    @torch.no_grad()
    def predict_batch(
        self,
        queries: List[str],
        documents: List[str]
    ) -> np.ndarray:
        """
        Predict relevance scores for batch of query-document pairs
        
        Returns:
            scores: [batch_size] relevance scores (0-1)
        """
        # Tokenize
        inputs = self.tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        scores = self.model.predict(**inputs)
        
        return scores.cpu().numpy()
    
    def evaluate(self) -> Dict[str, float]:
        """Main evaluation loop"""
        logger.info("\n" + "="*80)
        logger.info("STARTING EVALUATION")
        logger.info("="*80 + "\n")
        
        # Load everything
        self.load_model()
        self.load_corpus()
        test_df = self.load_test_data()
        
        # Prepare data
        logger.info("="*80)
        logger.info("PREDICTING SCORES")
        logger.info("="*80)
        
        all_predictions = []
        all_labels = []
        
        # Process in batches
        for i in tqdm(range(0, len(test_df), self.config.batch_size), desc="Evaluating"):
            batch_df = test_df.iloc[i:i + self.config.batch_size]
            
            # Prepare batch
            queries = batch_df['query'].tolist()
            labels = batch_df['label'].tolist()
            
            # Get document texts
            documents = []
            for _, row in batch_df.iterrows():
                if row['label'] == 1:
                    # Positive pair
                    doc_cid = int(row['positive_cid'])
                else:
                    # Negative pair
                    doc_cid = int(row['negative_cid']) if pd.notna(row.get('negative_cid')) else int(row['positive_cid'])
                
                doc_text = self.corpus.get(doc_cid, "")
                documents.append(doc_text)
            
            # Predict
            scores = self.predict_batch(queries, documents)
            
            all_predictions.extend(scores.tolist())
            all_labels.extend(labels)
        
        logger.info(f"✓ Predicted {len(all_predictions)} samples")
        
        # Compute metrics
        logger.info("="*80)
        logger.info("COMPUTING METRICS")
        logger.info("="*80)
        
        metrics = RerankMetrics.compute_all_metrics(
            predictions=all_predictions,
            labels=all_labels,
            threshold=0.5
        )
        
        # Print classification report
        RerankMetrics.print_classification_report(
            predictions=all_predictions,
            labels=all_labels,
            threshold=0.5
        )
        
        # Save results
        self.save_results(metrics, all_predictions, all_labels, test_df)
        
        return metrics
    
    def save_results(
        self,
        metrics: Dict[str, float],
        predictions: List[float],
        labels: List[int],
        test_df: pd.DataFrame
    ):
        """Save evaluation results"""
        logger.info("="*80)
        logger.info("SAVING RESULTS")
        logger.info("="*80)
        
        # Create results directory
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = self.config.results_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✓ Saved metrics: {metrics_path}")
        
        # Save predictions
        if self.config.save_predictions:
            predictions_df = test_df.copy()
            predictions_df['predicted_score'] = predictions
            predictions_df['predicted_label'] = (np.array(predictions) >= 0.5).astype(int)
            
            predictions_path = self.config.results_dir / "predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)
            logger.info(f"✓ Saved predictions: {predictions_path}")
        
        # Save failed cases (false positives and false negatives)
        if self.config.save_failed_cases:
            predictions_array = np.array(predictions)
            labels_array = np.array(labels)
            binary_preds = (predictions_array >= 0.5).astype(int)
            
            failed_indices = np.where(binary_preds != labels_array)[0]
            
            if len(failed_indices) > 0:
                failed_df = test_df.iloc[failed_indices].copy()
                failed_df['predicted_score'] = predictions_array[failed_indices]
                failed_df['predicted_label'] = binary_preds[failed_indices]
                failed_df['error_type'] = [
                    'False Positive' if pred == 1 else 'False Negative'
                    for pred in binary_preds[failed_indices]
                ]
                
                failed_path = self.config.results_dir / "failed_cases.csv"
                failed_df.to_csv(failed_path, index=False)
                logger.info(f"✓ Saved {len(failed_df)} failed cases: {failed_path}")
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Pretty print metrics"""
        logger.info("\n" + "="*80)
        logger.info("EVALUATION RESULTS")
        logger.info("="*80)
        
        logger.info("\n📊 Classification Metrics:")
        logger.info(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"  Precision:   {metrics['precision']:.4f}")
        logger.info(f"  Recall:      {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:    {metrics['f1']:.4f}")
        logger.info(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
        
        logger.info("\n📊 Confusion Matrix:")
        logger.info(f"  True Positives:  {metrics['true_positives']}")
        logger.info(f"  True Negatives:  {metrics['true_negatives']}")
        logger.info(f"  False Positives: {metrics['false_positives']}")
        logger.info(f"  False Negatives: {metrics['false_negatives']}")
        
        logger.info("\n📊 Additional Metrics:")
        logger.info(f"  Specificity: {metrics['specificity']:.4f}")
        logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        logger.info("")
    
    def run(self):
        """Run full evaluation pipeline"""
        try:
            # Setup MLflow
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            
            with mlflow.start_run():
                # Log config
                mlflow.log_params({
                    'mlflow_model_name': self.config.mlflow_model_name,
                    'mlflow_model_stage': self.config.mlflow_model_stage,
                    'test_path': str(self.config.test_path),
                    'batch_size': self.config.batch_size,
                    'max_seq_length': self.config.max_seq_length,
                    'device': self.config.device
                })
                
                # Evaluate
                metrics = self.evaluate()
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log artifacts
                mlflow.log_artifacts(str(self.config.results_dir))
                
                # Print results
                self.print_metrics(metrics)
                
                logger.info("="*80)
                logger.info("✓ EVALUATION COMPLETED!")
                logger.info("="*80)
                logger.info(f"\n📂 Results saved to: {self.config.results_dir}")
                
                return metrics
                
        except Exception as e:
            logger.error(f"✗ Evaluation failed: {e}")
            raise