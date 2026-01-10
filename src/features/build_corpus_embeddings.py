import torch
import pandas as pd
from tqdm import tqdm
from typing import Dict
from transformers import AutoTokenizer

from src.common.logging_config import get_logger
from src.models.bi_encoder.model import BiEncoder
from src.entity.config_entity import CorpusEmbeddingsConfig
# from src.common.s3_utils import S3Client

logger = get_logger(__name__)


class CorpusEmbeddingBuilder:
    """
    Build corpus embeddings for training/evaluation
    Save locally as .pt file for:
    - Hard negative mining
    - Bi-encoder evaluation
    - Cross-encoder training data generation
    """

    def __init__(self, config: CorpusEmbeddingsConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else " cpu"
        )

        self.model = None
        self.tokenizer = None

        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Max seq length: {self.config.max_seq_length}")

    def load_model(self):
        """Load trained bi-encoder model from MLflow"""
        logger.info("="*80)
        logger.info("LOADING MODEL FROM MLFLOW")
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
        
        # Download model artifacts from MLflow
        logger.info("Downloading model artifacts...")
        model_path = Path(mlflow.artifacts.download_artifacts(model_uri))
        
        # Debug: show structure
        logger.info(f"Model downloaded to: {model_path}")
        logger.info("Contents:")
        for item in model_path.rglob("*"):
            if item.is_file():
                logger.info(f"  {item.relative_to(model_path)}")
        
        # Try different possible locations
        possible_paths = [
            model_path / "best_model.pt",
            model_path / "pytorch_model.bin",
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
            # Full checkpoint with metadata
            model_config = checkpoint.get('config')
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # Alternative format
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
                
                # Create a simple config object
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

    def load_corpus(self) -> pd.DataFrame:
        """Load corpus data"""
        logger.info("="*80)
        logger.info("LOADING CORPUS")
        logger.info("="*80)
        logger.info(f"Corpus path: {self.config.corpus_path}")
        
        if not self.config.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.config.corpus_path}")
        
        df = pd.read_csv(self.config.corpus_path)
        
        # Validate columns
        required_cols = ['cid', 'text']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in corpus: {missing_cols}")
        
        # Remove duplicates and nulls
        original_len = len(df)
        df = df.dropna(subset=['cid', 'text'])
        df = df.drop_duplicates(subset=['cid'])
        
        if len(df) < original_len:
            logger.warning(f"Removed {original_len - len(df)} invalid/duplicate rows")
        
        logger.info(f"✓ Loaded {len(df)} documents")
        
        return df
    
    @torch.no_grad()
    def encode_corpus(self, corpus_df: pd.DataFrame) -> dict:
        """Encode all corpus documents"""
        logger.info("="*80)
        logger.info("ENCODING CORPUS")
        logger.info("="*80)

        texts = corpus_df['text'].to_list()
        cids = corpus_df['cid'].to_list()

        logger.info(f"Total documents: {len(texts)}")
        logger.info(f"Batch size: {self.config.batch_size}")
        num_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size
        logger.info(f"Total batches: {num_batches}")

        all_embeddings = []

        # Process in batches
        for i in tqdm(range(0, len(texts), self.config.batch_size), desc='Encoding'):
            batch_texts = texts[i:i + self.config.batch_size]

            # Tokenizer
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors='pt'
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Encode
            embeddings = self.model.encode(**inputs)

            # Move to CP to save memory
            all_embeddings.append(embeddings.cpu())
        
        # Concatenate all embeddings
        embeddings_tensor = torch.cat(all_embeddings, dim=0)

        logger.info("✓ Encoding complete")
        logger.info(f"  Embeddings shape: {embeddings_tensor.shape}")
        logger.info(f"  Embedding dimension: {embeddings_tensor.shape[1]}")

        return {
            'embeddings': embeddings_tensor,
            'cids': cids,
            'texts': texts
        }
    
    def save_embeddings(self, data: Dict):
        """Save embeddings to disk"""
        logger.info("="*80)
        logger.info("SAVING EMBEDDINGS")
        logger.info("="*80)
        
        output_path = self.config.output_dir / self.config.embeddings_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output path: {output_path}")
        
        # Save with metadata
        torch.save({
            'embeddings': data['embeddings'],
            'cids': data['cids'],
            'texts': data['texts'],
            'metadata': {
                'num_documents': len(data['cids']),
                'embedding_dim': data['embeddings'].shape[1],
                'mlflow_model_name': self.config.mlflow_model_name,
                'mlflow_model_stage': self.config.mlflow_model_stage,
                'corpus_path': str(self.config.corpus_path),
                'max_seq_length': self.config.max_seq_length,
                'batch_size': self.config.batch_size,
                'created_at': pd.Timestamp.now().isoformat(),
                'device': str(self.device)
            }
        }, output_path)
        
        # Calculate file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        logger.info("✓ Embeddings saved")
        logger.info(f"  File size: {file_size_mb:.2f} MB")

    def build(self):
        """Main pipeline: load model → load corpus → encode → save"""
        logger.info("\n" + "="*80)
        logger.info("BUILD CORPUS EMBEDDINGS PIPELINE")
        logger.info("="*80 + "\n")

        try:
            # Step 1: Load model
            self.load_model()

            # Step 2: Load corpus
            corpus_df = self.load_corpus()

            # Step 3: Encode corpus
            embedding_data = self.encode_corpus(corpus_df)

            # Step 4: Save embeddings
            self.save_embeddings(embedding_data)

            # Summary
            logger.info("\n" + "="*80)
            logger.info("✓ CORPUS EMBEDDINGS BUILD COMPLETED!")
            logger.info("="*80)
            output_path = self.config.output_dir / self.config.embeddings_file
            logger.info(f"\n📦 Output: {output_path}")
            logger.info("\n📝 Use this file for:")
            logger.info("  • Bi-encoder evaluation (Recall@K, MRR)")
            logger.info("  • Hard negative mining")
            logger.info("  • Cross-encoder training data generation")
            logger.info("")

        except Exception as e:
            logger.error(f"\n✗ Build failed: {e}")
            raise