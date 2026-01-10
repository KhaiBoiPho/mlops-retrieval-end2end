import json
from pathlib import Path
import shutil
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
import mlflow
from typing import List, Optional, Set

from src.entity.config_entity import CrossEncoderTrainingConfig
from src.models.cross_encoder.model import CrossEncoder
from src.common.logging_config import get_logger
from src.models.cross_encoder.loss import CrossEncoderLoss
from src.train.cross_encoder.dataset import create_cross_encoder_dataloader
from src.train.callbacks import EarlyStopping, MLflowMetricsLogger
from src.registry.mlflow_client import get_mlflow_client
from src.common.s3_utils import S3Client
from src.registry.model_registry import ModelRegistry

logger = get_logger(__name__)


class CrossEncoderTrainer:
    """Cross-Encoder training with pure PyTorch"""

    def __init__(self, config: CrossEncoderTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize MLflow
        self.mlflow_client = get_mlflow_client()

        # Track all saved files
        self.checkpoint_paths: List[Path] = []
        self.best_checkpoint_path: Optional[Path] = None
        self.all_saved_files: Set[Path] = set()

        self.model: Optional[CrossEncoder] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self.criterion: Optional[nn.Module] = None

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        self.global_step = 0
        self.best_val_loss = float('inf')

        self.early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        self.mlflow_logger = MLflowMetricsLogger(log_every_n_steps=100)

    def prepare_data(self):
        """Load and prepare datasets"""
        logger.info("="*80)
        logger.info("PREPARING DATA")
        logger.info("="*80)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Get max_samples from config (if exists)
        max_samples = getattr(self.config, 'max_samples', None)

        # Create dataloaders
        self.train_loader = create_cross_encoder_dataloader(
            data_path=self.config.train_data_path,
            corpus_path=self.config.corpus_path,
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            max_length=self.config.max_seq_length,
            shuffle=True,
            num_workers=4,
            max_samples=max_samples
        )

        self.val_loader = create_cross_encoder_dataloader(
            data_path=self.config.val_data_path,
            corpus_path=self.config.corpus_path,
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            max_length=self.config.max_seq_length,
            shuffle=False,
            num_workers=4,
            max_samples=max_samples // 5 if max_samples else None
        )

        logger.info(f"✓ Train batches: {len(self.train_loader)}")
        logger.info(f"✓ Val batches: {len(self.val_loader)}")

    def build_model(self):
        """Initialize model, optimizer, loss"""
        logger.info("="*80)
        logger.info("BUILDING MODEL")
        logger.info("="*80)

        # Model
        self.model = CrossEncoder(
            model_name=self.config.model_name,
            num_labels=self.config.num_labels
        ).to(self.device)

        # Loss function
        self.criterion = CrossEncoderLoss()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if self.config.use_fp16 else None

        logger.info(f"✓ Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"✓ Total training steps: {total_steps}")
        logger.info(f"✓ Warmup steps: {warmup_steps}")

    def train_epoch(self, epoch: int):
        """Train one epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.config.use_fp16:
                with torch.amp.autocast(device_type='cuda'):
                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    loss = self.criterion(logits, labels)
                
                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                loss = self.criterion(logits, labels)
                
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

            self.mlflow_logger.on_step(self.global_step, {
                'train_loss': loss.item(),
                'learning_rate': self.scheduler.get_last_lr()[0]
            })

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0

        for batch in tqdm(self.val_loader, desc='Validation'):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss = self.criterion(logits, labels)

            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        Save checkpoint and upload to S3 immediately if incremental strategy
        """
        
        # 1. Save standard checkpoint
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__
        }
        
        checkpoint_path = self.config.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_paths.append(checkpoint_path)
        self.all_saved_files.add(checkpoint_path)
        
        logger.info(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # 2. If best model, save in serving format
        if is_best:
            logger.info("💎 Saving best model...")
            
            # Remove old best model folder if exists
            best_model_dir = self.config.output_dir / "best_model"
            if best_model_dir.exists():
                shutil.rmtree(best_model_dir)
            
            best_model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model weights
            model_weights_path = best_model_dir / "pytorch_model.bin"
            torch.save(self.model.state_dict(), model_weights_path)
            self.all_saved_files.add(model_weights_path)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(best_model_dir)
            
            # Track tokenizer files
            for file in best_model_dir.glob("*"):
                if file.is_file():
                    self.all_saved_files.add(file)
            
            # Save model config with metadata
            model_config = {
                'model_name': self.config.model_name,
                'num_labels': self.config.num_labels,
                'max_seq_length': self.config.max_seq_length,
                'best_val_loss': float(val_loss),
                'epoch': epoch,
                'global_step': self.global_step,
                'training_config': {
                    k: str(v) if isinstance(v, Path) else v 
                    for k, v in self.config.__dict__.items()
                }
            }
            
            config_path = best_model_dir / "model_config.json"
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2, default=str)
            self.all_saved_files.add(config_path)
            
            # Save best checkpoint .pt file
            best_checkpoint_path = self.config.output_dir / "best_model.pt"
            torch.save(checkpoint, best_checkpoint_path)
            self.best_checkpoint_path = best_checkpoint_path
            self.all_saved_files.add(best_checkpoint_path)
            
            # Log to MLflow
            mlflow.log_artifacts(str(best_model_dir), artifact_path="best_model")
            
            logger.info(f"✓ Best model saved to: {best_model_dir}")
        
        # 3. Upload to S3 incrementally if enabled
        if self.config.s3_enabled and self.config.upload_strategy == "incremental":
            run = mlflow.active_run()
            if run:
                self._upload_checkpoint_to_s3(checkpoint_path, run.info.run_id)
                
                if is_best:
                    self._upload_best_model_to_s3(run.info.run_id)
    
    def _upload_checkpoint_to_s3(self, checkpoint_path: Path, run_id: str):
        """Upload a single checkpoint file to S3"""
        if not self.config.s3_enabled:
            return
        
        from src.common.s3_utils import S3Client
        s3_client = S3Client()
        
        # S3 path: s3://bucket/models/cross-encoder/{run_id}/checkpoint_epoch_N.pt
        relative_path = checkpoint_path.relative_to(self.config.output_dir)
        s3_key = f"{self.config.s3_prefix}/{run_id}/{relative_path}"
        s3_key = s3_key.replace("\\", "/")
        
        try:
            s3_client.upload_file(
                local_path=checkpoint_path,
                s3_bucket=self.config.s3_bucket,
                s3_key=s3_key,
                show_progress=True
            )
            logger.info(f"  ☁️  Uploaded to S3: {s3_key}")
        except Exception as e:
            logger.error(f"  ❌ S3 upload failed for {checkpoint_path.name}: {e}")
    
    def _upload_best_model_to_s3(self, run_id: str):
        """Upload best_model/ directory to S3"""
        if not self.config.s3_enabled:
            return
        
        s3_client = S3Client()
        
        best_model_dir = self.config.output_dir / "best_model"
        
        if not best_model_dir.exists():
            return
        
        logger.info("  ☁️  Uploading best_model/ to S3...")
        
        # Upload all files in best_model/
        for file_path in best_model_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.config.output_dir)
                s3_key = f"{self.config.s3_prefix}/{run_id}/{relative_path}"
                s3_key = s3_key.replace("\\", "/")
                
                try:
                    s3_client.upload_file(
                        local_path=file_path,
                        s3_bucket=self.config.s3_bucket,
                        s3_key=s3_key,
                        show_progress=False
                    )
                except Exception as e:
                    logger.error(f"  ❌ Failed to upload {file_path.name}: {e}")
        
        # Also upload best_model.pt
        best_pt = self.config.output_dir / "best_model.pt"
        if best_pt.exists():
            s3_key = f"{self.config.s3_prefix}/{run_id}/best_model.pt"
            try:
                s3_client.upload_file(
                    local_path=best_pt,
                    s3_bucket=self.config.s3_bucket,
                    s3_key=s3_key,
                    show_progress=True
                )
            except Exception as e:
                logger.error(f"  ❌ Failed to upload best_model.pt: {e}")
    
    def upload_all_to_s3(self, run_id: str):
        """
        Upload ALL files in output_dir to S3
        Called at end of training if upload_strategy == "final"
        """
        if not self.config.s3_enabled:
            logger.info("⏭️  S3 upload disabled")
            return
        
        logger.info("="*80)
        logger.info("UPLOADING ALL ARTIFACTS TO S3")
        logger.info("="*80)
        
        s3_client = S3Client()
        
        # Get all files in output_dir
        output_dir = self.config.output_dir
        all_files = list(output_dir.rglob("*"))
        files_to_upload = [f for f in all_files if f.is_file()]
        
        logger.info(f"📦 Found {len(files_to_upload)} files to upload")
        logger.info(f"📍 Destination: s3://{self.config.s3_bucket}/{self.config.s3_prefix}/{run_id}/")
        
        # Upload with progress bar
        uploaded_count = 0
        failed_count = 0
        
        with tqdm(total=len(files_to_upload), desc="Uploading to S3") as pbar:
            for file_path in files_to_upload:
                try:
                    # Calculate relative path from output_dir
                    relative_path = file_path.relative_to(output_dir)
                    
                    # Create S3 key
                    s3_key = f"{self.config.s3_prefix}/{run_id}/{relative_path}"
                    s3_key = s3_key.replace("\\", "/")
                    
                    # Upload
                    s3_client.upload_file(
                        local_path=file_path,
                        s3_bucket=self.config.s3_bucket,
                        s3_key=s3_key,
                        show_progress=False
                    )
                    
                    uploaded_count += 1
                    pbar.set_postfix({
                        'uploaded': uploaded_count,
                        'failed': failed_count,
                        'current': file_path.name[:30]
                    })
                    
                except Exception as e:
                    failed_count += 1
                    logger.error(f"❌ Failed to upload {file_path.name}: {e}")
                
                pbar.update(1)
        
        logger.info(f"✓ Upload complete: {uploaded_count} succeeded, {failed_count} failed")
        logger.info(f"✓ S3 location: s3://{self.config.s3_bucket}/{self.config.s3_prefix}/{run_id}/")
        
        # Log S3 path to MLflow
        mlflow.log_param("s3_artifacts_path", f"s3://{self.config.s3_bucket}/{self.config.s3_prefix}/{run_id}")
    
    def register_model_to_registry(self, run_id: str):
        """Register best model to MLflow Model Registry"""
        if not self.config.register_model:
            logger.info("⏭️  Model registration disabled")
            return
        
        logger.info("="*80)
        logger.info("REGISTERING MODEL TO MLFLOW REGISTRY")
        logger.info("="*80)
        
        registry = ModelRegistry(self.config.mlflow_tracking_uri)
        
        try:
            model_version = registry.register_model(
                run_id=run_id,
                model_name=self.config.model_registry_name,
                model_path="best_model",
                tags={
                    'framework': 'pytorch',
                    'task': 'cross-encoder',
                    'dataset': 'legal-retrieval',
                    'best_val_loss': f"{self.best_val_loss:.4f}",
                    'num_epochs': str(self.config.num_epochs),
                    'model_base': self.config.model_name,
                    's3_backup': 'true' if self.config.s3_enabled else 'false'
                },
                description=f"Cross-Encoder for legal document re-ranking. "
                           f"Trained for {self.config.num_epochs} epochs. "
                           f"Best validation loss: {self.best_val_loss:.4f}"
            )
            
            logger.info(f"✓ Registered model: {self.config.model_registry_name}")
            logger.info(f"  Version: {model_version}")
            logger.info(f"  Run ID: {run_id}")
            
        except Exception as e:
            logger.error(f"❌ Model registration failed: {e}")
    
    def train(self):
        """Main training loop with full checkpoint persistence"""
        
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"🚀 MLflow Run ID: {run_id}")
            
            # Log parameters
            mlflow.log_params({
                "model_name": self.config.model_name,
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "max_seq_length": self.config.max_seq_length,
                "loss_type": self.config.loss_type,
                "warmup_ratio": self.config.warmup_ratio,
                "weight_decay": self.config.weight_decay,
                "s3_enabled": self.config.s3_enabled,
                "upload_strategy": self.config.upload_strategy if self.config.s3_enabled else "none"
            })
            
            # Prepare data & build model
            self.prepare_data()
            self.build_model()
            
            logger.info("="*80)
            logger.info("STARTING TRAINING")
            logger.info("="*80)
            
            # Training loop
            for epoch in range(self.config.num_epochs):
                train_loss = self.train_epoch(epoch)
                val_loss = self.validate()
                
                logger.info(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
                logger.info(f"  Train Loss: {train_loss:.4f}")
                logger.info(f"  Val Loss:   {val_loss:.4f}")
                
                self.mlflow_logger.on_epoch_end(epoch, {
                    'epoch_train_loss': train_loss,
                    'epoch_val_loss': val_loss
                })
                
                # Check if best
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    logger.info("  💎 New best model!")
                
                # Save checkpoint (uploads to S3 if incremental)
                self.save_checkpoint(epoch, val_loss, is_best=is_best)
                
                # Early stopping
                if self.early_stopping(val_loss):
                    logger.info(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Post-training steps
            logger.info("="*80)
            logger.info("POST-TRAINING PROCESSING")
            logger.info("="*80)
            
            # Upload all to S3 if final strategy
            if self.config.s3_enabled and self.config.upload_strategy == "final":
                self.upload_all_to_s3(run_id)
            
            # Register model to MLflow Registry
            self.register_model_to_registry(run_id)
            
            # Save DVC metrics
            self._save_dvc_metrics(run_id)
            
            # Final summary
            logger.info("="*80)
            logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"  Best Val Loss: {self.best_val_loss:.4f}")
            logger.info(f"  Run ID: {run_id}")
            logger.info(f"  Local artifacts: {self.config.output_dir}")
            logger.info(f"  Total checkpoints: {len(self.checkpoint_paths)}")
            if self.config.s3_enabled:
                logger.info(f"  S3 backup: s3://{self.config.s3_bucket}/{self.config.s3_prefix}/{run_id}/")
            logger.info("="*80)
            
            return run_id
    
    def _save_dvc_metrics(self, run_id: str):
        """Save simple metrics for DVC tracking"""
        
        dvc_metrics = {
            "best_val_loss": float(self.best_val_loss),
            "mlflow_run_id": run_id,
            "num_checkpoints": len(self.checkpoint_paths),
            "status": "completed"
        }
        
        metrics_path = self.config.output_dir / "dvc_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(dvc_metrics, f, indent=2)
        
        logger.info(f"✓ Saved DVC metrics: {metrics_path}")