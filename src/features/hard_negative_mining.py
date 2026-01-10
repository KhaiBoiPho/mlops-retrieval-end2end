import torch
import pandas as pd
from tqdm import tqdm
from typing import List
import ast
from transformers import AutoTokenizer

from src.common.logging_config import get_logger
from src.models.bi_encoder.model import BiEncoder
from src.entity.config_entity import HardNegativeMiningConfig
# from src.common.s3_utils import S3Client

logger = get_logger(__name__)


class HardNegativeMiner:
    """
    Mine hard negatives for cross-encoder training
    
    Process:
    1. For each training query
    2. Retrieve top-K similar documents (candidates)
    3. Filter out positive documents
    4. Sample N hard negatives
    5. Create (query, positive, negative) pairs
    """
    
    def __init__(self, config: HardNegativeMiningConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        
        self.model = None
        self.tokenizer = None
        self.corpus_embeddings = None
        self.corpus_cids = None
        self.corpus_texts = None
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Top-K candidates: {self.config.top_k_candidates}")
        logger.info(f"Negatives per query: {self.config.num_negatives_per_query}")
    
    def load_model(self):
        """Load trained bi-encoder model from MLflow"""
        logger.info("="*80)
        logger.info("LOADING BI-ENCODER MODEL FROM MLFLOW")
        logger.info("="*80)
        logger.info(f"Model name: {self.config.mlflow_model_name}")
        logger.info(f"Model stage: {self.config.mlflow_model_stage}")
        
        import mlflow
        from pathlib import Path
        
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        
        # Build model URI
        if self.config.mlflow_run_id:
            model_uri = f"runs:/{self.config.mlflow_run_id}/best_model"
            logger.info(f"Loading from run ID: {self.config.mlflow_run_id}")
        elif self.config.mlflow_model_stage.lower() == "latest":
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
        
        # Download model artifacts
        logger.info("Downloading model artifacts...")
        model_path = Path(mlflow.artifacts.download_artifacts(model_uri))
        
        # Try different possible locations
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
            state_dict = checkpoint
            model_config = None
        
        if model_config is None:
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
                raise ValueError("Model config not found")
        
        self.model = BiEncoder(
            model_name=model_config.model_name,
            pooling=getattr(model_config, 'pooling', 'mean'),
            normalize=getattr(model_config, 'normalize', True)
        )
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model.encoder.config._name_or_path
        )
        
        logger.info(f"✓ Model loaded: {model_config.model_name}")
        
    def load_corpus_embeddings(self):
        """Load pre-computed corpus embeddings"""
        logger.info("="*80)
        logger.info("LOADING CORPUS EMBEDDINGS")
        logger.info("="*80)
        
        if not self.config.embeddings_path.exists():
            raise FileNotFoundError(
                f"Corpus embeddings not found: {self.config.embeddings_path}\n"
                f"Please run: python dags/build_corpus_embeddings.py"
            )
        
        data = torch.load(self.config.embeddings_path, weights_only=False)
        
        # Keep on CPU to save memory
        self.corpus_embeddings = data['embeddings']
        self.corpus_cids = data['cids']
        self.corpus_texts = data['texts']
        
        logger.info("✓ Loaded corpus embeddings")
        logger.info(f"  Shape: {self.corpus_embeddings.shape}")
        logger.info(f"  Num documents: {len(self.corpus_cids)}")
    
    def load_train_data(self) -> pd.DataFrame:
        """Load training data"""
        logger.info("="*80)
        logger.info("LOADING INPUT DATA")
        logger.info("="*80)
        
        if not self.config.input_path.exists():
            raise FileNotFoundError(f"Input data not found: {self.config.input_path}")
        
        df = pd.read_csv(self.config.input_path)
        
        required_cols = ['question', 'cid', 'context']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        logger.info(f"✓ Loaded {len(df)} queries")
        
        return df
    
    def parse_positive_cids(self, cid_value) -> List[int]:
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
            
            inputs = self.tokenizer(
                batch_queries,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embeddings = self.model.encode(**inputs)
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def retrieve_candidates(
        self,
        query_embeddings: torch.Tensor
    ) -> tuple:
        """
        Retrieve top-K candidates for each query
        
        Returns:
            scores: [num_queries, k] similarity scores
            indices: [num_queries, k] document indices
        """
        # Compute similarities in batches to save memory
        num_queries = query_embeddings.shape[0]
        all_scores = []
        all_indices = []
        
        batch_size = 100
        
        for i in tqdm(range(0, num_queries, batch_size), desc="Retrieving candidates"):
            query_batch = query_embeddings[i:i + batch_size]
            
            # Compute similarity: [batch_size, num_docs]
            similarities = torch.matmul(query_batch, self.corpus_embeddings.T)
            
            # Get top-k
            top_k_scores, top_k_indices = torch.topk(
                similarities,
                k=self.config.top_k_candidates,
                dim=1
            )
            
            all_scores.append(top_k_scores)
            all_indices.append(top_k_indices)
        
        final_scores = torch.cat(all_scores, dim=0)
        final_indices = torch.cat(all_indices, dim=0)
        
        return final_scores, final_indices
    
    def mine_hard_negatives(
        self,
        train_df: pd.DataFrame,
        candidate_scores: torch.Tensor,
        candidate_indices: torch.Tensor
    ) -> pd.DataFrame:
        """
        Mine hard negatives for each query
        
        Returns:
            DataFrame with (query, positive_cid, negative_cid, label)
        """
        logger.info("="*80)
        logger.info("MINING HARD NEGATIVES")
        logger.info("="*80)
        
        data_rows = []
        
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Mining"):
            query = row['question']
            positive_cids = self.parse_positive_cids(row['cid'])
            positive_cids_set = set(positive_cids)
            
            # Get candidates for this query
            scores = candidate_scores[idx].numpy()
            indices = candidate_indices[idx].numpy()
            
            # Filter hard negatives
            hard_negatives = []
            for score, doc_idx in zip(scores, indices):
                candidate_cid = self.corpus_cids[doc_idx]
                
                # Skip if it's a positive
                if candidate_cid in positive_cids_set:
                    continue
                
                # Skip if score too low
                if score < self.config.min_score_threshold:
                    continue
                
                hard_negatives.append((candidate_cid, score))
            
            # Sample N hard negatives
            if len(hard_negatives) > self.config.num_negatives_per_query:
                # Sort by score (descending) and take top N
                hard_negatives.sort(key=lambda x: x[1], reverse=True)
                hard_negatives = hard_negatives[:self.config.num_negatives_per_query]
            
            # Create positive pairs
            for pos_cid in positive_cids:
                data_rows.append({
                    'query': query,
                    'positive_cid': pos_cid,
                    'negative_cid': None,
                    'label': 1
                })
            
            # Create negative pairs
            for neg_cid, neg_score in hard_negatives:
                # Pair with first positive (or random positive)
                pos_cid = positive_cids[0]
                data_rows.append({
                    'query': query,
                    'positive_cid': pos_cid,
                    'negative_cid': neg_cid,
                    'label': 0
                })
        
        result_df = pd.DataFrame(data_rows)
        
        logger.info("✓ Mined hard negatives")
        logger.info(f"  Total pairs: {len(result_df)}")
        logger.info(f"  Positive pairs: {(result_df['label'] == 1).sum()}")
        logger.info(f"  Negative pairs: {(result_df['label'] == 0).sum()}")
        
        return result_df
    
    def save_dataset(self, df: pd.DataFrame):
        """Save cross-encoder training dataset"""
        logger.info("="*80)
        logger.info("SAVING DATASET")
        logger.info("="*80)
        
        df.to_csv(self.config.output_path, index=False)
        
        file_size_mb = self.config.output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Saved to: {self.config.output_path}")
        logger.info(f"  File size: {file_size_mb:.2f} MB")
        logger.info(f"  Total pairs: {len(df)}")
    
    def run(self):
        """Run full hard negative mining pipeline"""
        logger.info("\n" + "="*80)
        logger.info("HARD NEGATIVE MINING PIPELINE")
        logger.info("="*80 + "\n")
        
        try:
            # Load model and data
            self.load_model()
            self.load_corpus_embeddings()
            train_df = self.load_train_data()
            
            # Encode queries
            logger.info("="*80)
            logger.info("ENCODING QUERIES")
            logger.info("="*80)
            
            queries = train_df['question'].tolist()
            logger.info(f"Encoding {len(queries)} queries...")
            query_embeddings = self.encode_queries(queries)
            logger.info(f"✓ Query embeddings: {query_embeddings.shape}")
            
            # Retrieve candidates
            candidate_scores, candidate_indices = self.retrieve_candidates(query_embeddings)
            
            # Mine hard negatives
            cross_encoder_df = self.mine_hard_negatives(
                train_df,
                candidate_scores,
                candidate_indices
            )
            
            # Save dataset
            self.save_dataset(cross_encoder_df)
            
            # Summary
            logger.info("\n" + "="*80)
            logger.info("✓ HARD NEGATIVE MINING COMPLETED!")
            logger.info("="*80)
            logger.info(f"\n📦 Output: {self.config.output_path}")
            logger.info("\n📊 Dataset Statistics:")
            logger.info(f"  Total pairs: {len(cross_encoder_df)}")
            logger.info(f"  Positive: {(cross_encoder_df['label'] == 1).sum()}")
            logger.info(f"  Negative: {(cross_encoder_df['label'] == 0).sum()}")
            logger.info("\n🎯 Next: Train cross-encoder\n")
            
        except Exception as e:
            logger.error(f"\n✗ Mining failed: {e}")
            raise