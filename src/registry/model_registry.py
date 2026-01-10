import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Optional

class ModelRegistry:
    """Centralized Model Registry Management"""
    
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        model_path: str = "best_model",  # artifact path in MLflow
        tags: Optional[Dict] = None,
        description: str = ""
    ) -> str:
        """
        Register model to MLflow Model Registry
        
        Returns:
            version: Model version string
        """
        # Build model URI
        model_uri = f"runs:/{run_id}/{model_path}"
        
        # Register model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags=tags
        )
        
        # Add description and tags
        if description:
            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
        
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=key,
                    value=str(value)
                )
        
        return model_version.version
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,  # "Staging", "Production", "Archived"
        archive_existing: bool = True
    ):
        """Transition model to different stage"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
    
    def get_production_model(self, model_name: str):
        """Get current production model version"""
        versions = self.client.get_latest_versions(
            model_name, 
            stages=["Production"]
        )
        return versions[0] if versions else None
    
    def compare_models(
        self,
        model_name: str,
        version_1: str,
        version_2: str,
        metrics: list
    ) -> Dict:
        """Compare metrics between two model versions"""
        results = {}
        
        for version in [version_1, version_2]:
            model_version = self.client.get_model_version(model_name, version)
            run = self.client.get_run(model_version.run_id)
            results[version] = {
                metric: run.data.metrics.get(metric)
                for metric in metrics
            }
        
        return results