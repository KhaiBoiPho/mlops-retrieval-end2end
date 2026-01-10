from pathlib import Path
import torch
import pandas as pd
from typing import Dict, List, Tuple
import ast
import mlflow
from transformers import AutoTokenizer

from src.common.logging_config import get_logger
from src.models.bi_encoder.model import BiEncoder
from src.entity.config_entity import BiEncoderEvalConfig
from src.eval.metrics.retrieval_metrics import RetrievalMetrics
from src.registry.mlflow_client import get_mlflow_client

logger = get_logger(__name__)


class BiEncoderEvaluator:
    """
    Evaluate bi-encoder model on retrieval task
    Computes: Recall@K, Precision@K, MRR, MAP, NDCG@K, Hit Rate@K
    """

    def __init__(self, config: BiEncoderEvalConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        self.mlflow_client = get_mlflow_client()

        self.model = None
        self.tokenizer = None
        self.corpus_embeddings = None
        self.corpus_cids = None

        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.config.batch_size}")

    def load_model(self):
        """Load trained bi-encoder model from MLflow"""
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
                import json
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
        self.model = BiEncoder(
            model_name=model_config.model_name,
            pooling=getattr(model_config, 'pooling', 'mean'),
            normalize=getattr(model_config, 'normalize', True)
        )
        
        # Load state dict
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model.encoder.config._name_or_path
        )
        
        logger.info(f"✓ Model loaded: {model_config.model_name}")
        logger.info(f"  Pooling: {self.model.pooling}")
        logger.info(f"  Normalize: {self.model.normalize}")
        logger.info(f"  Hidden size: {self.model.hidden_size}")

    def load_corpus_embeddings(self):
        """Load pre-computed corpus embeddings"""
        logger.info("="*80)
        logger.info("LOADING CORPUS EMBEDDINGS")
        logger.info("="*80)
        logger.info(f"Path: {self.config.corpus_embeddings_path}")
        
        if not self.config.corpus_embeddings_path.exists():
            raise FileNotFoundError(
                f"Corpus embeddings not found: {self.config.corpus_embeddings_path}\n"
                f"Please run: python dags/build_corpus_embeddings.py"
            )
        
        data = torch.load(
            self.config.corpus_embeddings_path,
            weights_only=False
        )
        
        self.corpus_embeddings = data['embeddings'].to(self.device)
        self.corpus_cids = data['cids']
        
        logger.info("✓ Loaded corpus embeddings")
        logger.info(f"  Shape: {self.corpus_embeddings.shape}")
        logger.info(f"  Num documents: {len(self.corpus_cids)}")

    def load_test_data(self) -> pd.DataFrame:
        """Load test data"""
        logger.info("="*80)
        logger.info("LOADING TEST DATA")
        logger.info("="*80)
        logger.info(f"Test path: {self.config.test_path}")
        
        if not self.config.test_path.exists():
            raise FileNotFoundError(f"Test data not found: {self.config.test_path}")
        
        df = pd.read_csv(self.config.test_path)

        # # Sample 10% for quick testing
        # original_size = len(df)
        # df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
        
        # logger.warning(f"⚠ Using 10% sample for testing: {len(df)}/{original_size} queries")
        
        # Validate columns
        required_cols = ['question', 'cid']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in test data: {missing_cols}")
        
        logger.info(f"✓ Loaded {len(df)} test queries")
        
        return df
    
    def parse_ground_truth_cids(self, cid_value) -> List[int]:
        """Parse CID column (can be int or list)"""
        if isinstance(cid_value, str) and cid_value.startswith('['):
            try:
                cids = ast.literal_eval(cid_value)
                return [int(c) for c in cids]
            except Exception:
                return [int(cid_value)]
        else:
            return [int(cid_value)]
        
    @torch.no_grad()
    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        """Encode batch of queries"""
        all_embeddings = []

        for i in range(0, len(queries), self.config.batch_size):
            batch_queries = queries[i:i + self.config.batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_queries,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors='pt'
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Encode
            embeddings = self.model.encode(**inputs)
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)
    
    def retrieve_top_k(
        self,
        query_embeddings: torch.Tensor,
        k: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k documents for each query
        
        Returns:
            scores: [num_queries, k] similarity scores
            indices: [num_queries, k] document indices
        """
        # Compute similarity: [num_queries, num_docs]
        similarities = torch.matmul(query_embeddings, self.corpus_embeddings.T)

        # Get top-k
        top_k_scores, top_k_indices = torch.topk(similarities, k=k, dim=1)

        return top_k_scores, top_k_indices
    
    def evaluate(self) -> Dict[str, float]:
        """Main evaluation loop"""
        logger.info("\n" + "="*80)
        logger.info("STARTING EVALUATION")
        logger.info("="*80 + "\n")

        # Load everything
        self.load_model()
        self.load_corpus_embeddings()
        test_df = self.load_test_data()

        # Prepare data
        logger.info("="*80)
        logger.info("ENCODING QUERIES")
        logger.info("="*80)

        queries = test_df['question'].tolist()
        ground_truth_cids = [
            self.parse_ground_truth_cids(cid)
            for cid in test_df['cid'].to_list()
        ]

        # Encode queries
        logger.info(f"Encoding {len(queries)} queries ...")
        query_embeddings = self.encode_queries(queries)
        logger.info(f"✓ Query embeddings: {query_embeddings.shape}")

        # Retrieve top-k
        logger.info("="*80)
        logger.info("RETRIEVING TOP-K DOCUMENTS")
        logger.info("="*80)

        max_k = max(
            max(self.config.recall_k_values),
            max(self.config.precision_k_values),
            max(self.config.ndcg_k_values)
        )

        logger.info(f"Retrieving top-{max_k} documents for each query...")
        top_k_scores, top_k_indices = self.retrieve_top_k(query_embeddings, k=max_k)
        
        # Convert indices to CIDs
        predictions = []
        for indices in top_k_indices.cpu().numpy():
            pred_cids = [self.corpus_cids[idx] for idx in indices]
            predictions.append(pred_cids)
        
        logger.info(f"✓ Retrieved top-{max_k} for {len(predictions)} queries")
        
        # Compute metrics
        logger.info("="*80)
        logger.info("COMPUTING METRICS")
        logger.info("="*80)
        
        metrics = RetrievalMetrics.compute_all_metrics(
            predictions=predictions,
            ground_truths=ground_truth_cids,
            recall_k=self.config.recall_k_values,
            precision_k=self.config.precision_k_values,
            ndcg_k=self.config.ndcg_k_values
        )
        
        # Save results
        self.save_results(metrics, predictions, ground_truth_cids, queries, top_k_scores)
        
        return metrics
    
    def save_results(
        self,
        metrics: Dict[str, float],
        predictions: List[List[int]],
        ground_truths: List[List[int]],
        queries: List[str],
        scores: torch.Tensor
    ):
        """Save evaluation results"""
        logger.info("="*80)
        logger.info("SAVING RESULTS")
        logger.info("="*80)
        
        # Save metrics to JSON
        import json
        metrics_path = self.config.results_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✓ Saved metrics: {metrics_path}")
        
        # Save predictions
        if self.config.save_predictions:
            predictions_df = pd.DataFrame({
                'query': queries,
                'ground_truth': ground_truths,
                'predictions': predictions
            })
            predictions_path = self.config.results_dir / "predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)
            logger.info(f"✓ Saved predictions: {predictions_path}")
        
        # Save failed cases (queries with no hits in top-K)
        if self.config.save_failed_cases:
            failed_cases = []
            for i, (pred, truth) in enumerate(zip(predictions, ground_truths)):
                # Check top-100 (or max available)
                if not any(doc_id in truth for doc_id in pred[:100]):
                    failed_cases.append({
                        'query_idx': i,
                        'query': queries[i],
                        'ground_truth_cids': truth,
                        'top_10_predictions': pred[:10]
                    })
            
            if failed_cases:
                failed_df = pd.DataFrame(failed_cases)
                failed_path = self.config.results_dir / "failed_cases.csv"
                failed_df.to_csv(failed_path, index=False)
                logger.info(f"✓ Saved {len(failed_cases)} failed cases: {failed_path}")
        
        # Save rankings (top-K with scores)
        if self.config.save_rankings:
            rankings = []
            for i in range(min(100, len(queries))):  # Save first 100 queries
                rankings.append({
                    'query': queries[i],
                    'ground_truth': ground_truths[i],
                    'top_10_cids': predictions[i][:10],
                    'top_10_scores': scores[i][:10].cpu().numpy().tolist()
                })
            
            rankings_df = pd.DataFrame(rankings)
            rankings_path = self.config.results_dir / "rankings_sample.csv"
            rankings_df.to_csv(rankings_path, index=False)
            logger.info(f"✓ Saved rankings: {rankings_path}")
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Pretty print metrics"""
        logger.info("\n" + "="*80)
        logger.info("EVALUATION RESULTS")
        logger.info("="*80)
        
        # Recall metrics
        logger.info("\n📊 Recall@K:")
        for k in self.config.recall_k_values:
            logger.info(f"  Recall@{k:3d}: {metrics[f'recall@{k}']:.4f} ({metrics[f'recall@{k}']*100:.2f}%)")
        
        # Precision metrics
        logger.info("\n📊 Precision@K:")
        for k in self.config.precision_k_values:
            logger.info(f"  Precision@{k:3d}: {metrics[f'precision@{k}']:.4f}")
        
        # NDCG metrics
        logger.info("\n📊 NDCG@K:")
        for k in self.config.ndcg_k_values:
            logger.info(f"  NDCG@{k:3d}: {metrics[f'ndcg@{k}']:.4f}")
        
        # Hit Rate metrics
        logger.info("\n📊 Hit Rate@K:")
        for k in self.config.recall_k_values:
            logger.info(f"  Hit Rate@{k:3d}: {metrics[f'hit_rate@{k}']:.4f} ({metrics[f'hit_rate@{k}']*100:.2f}%)")
        
        # Overall metrics
        logger.info("\n📊 Overall Metrics:")
        logger.info(f"  MRR (Mean Reciprocal Rank): {metrics['mrr']:.4f}")
        logger.info(f"  MAP (Mean Average Precision): {metrics['map']:.4f}")
        
        logger.info("")
    
    def run(self):
        """Run full evaluation pipeline"""
        try:
            # Setup MLflow
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            
            with mlflow.start_run():
                # Log config (sửa đổi phần này)
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
                mlflow_metrics = {
                    key.replace('@', '_at_'): value 
                    for key, value in metrics.items()
                }
                mlflow.log_metrics(mlflow_metrics)
                
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