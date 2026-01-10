"""
scripts/deploy_runpod_api.py

Programmatically create RunPod templates and endpoints
"""
import os
import runpod
from typing import Dict

# Configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
DOCKER_REGISTRY = "your-dockerhub-username"  # Change this!

BI_ENCODER_CONFIG = {
    "name": "legal-bi-encoder",
    "image": f"{DOCKER_REGISTRY}/legal-bi-encoder:latest",
    "docker_args": "",
    "container_disk_in_gb": 10,
    "volume_in_gb": 0,
    "volume_mount_path": "",
    "ports": "8000/http",
    "env": {
        "MLFLOW_TRACKING_URI": "http://your-mlflow-server:5000",
        "MODEL_NAME": "bi-encoder",
        "MODEL_STAGE": "Production"
    }
}

CROSS_ENCODER_CONFIG = {
    "name": "legal-cross-encoder",
    "image": f"{DOCKER_REGISTRY}/legal-cross-encoder:latest",
    "docker_args": "",
    "container_disk_in_gb": 10,
    "volume_in_gb": 0,
    "volume_mount_path": "",
    "ports": "8001/http",
    "env": {
        "MLFLOW_TRACKING_URI": "http://your-mlflow-server:5000",
        "MODEL_NAME": "cross-encoder",
        "MODEL_STAGE": "Production"
    }
}


def create_template(config: Dict) -> str:
    """Create RunPod template"""
    print(f"\n{'='*60}")
    print(f"Creating template: {config['name']}")
    print(f"{'='*60}")
    
    try:
        # Note: RunPod Python SDK doesn't have direct template creation
        # You'll need to use their REST API or create via UI
        # This is a placeholder for the structure
        
        print("✓ Template config prepared:")
        print(f"  Image: {config['image']}")
        print(f"  Container Disk: {config['container_disk_in_gb']} GB")
        print(f"  Ports: {config['ports']}")
        print(f"  Environment Variables: {len(config['env'])} vars")
        
        # TODO: Implement actual API call when SDK supports it
        # For now, return placeholder
        template_id = f"template-{config['name']}"
        
        return template_id
        
    except Exception as e:
        print(f"❌ Error creating template: {e}")
        raise


def create_endpoint(
    template_id: str,
    name: str,
    gpu_ids: str = "AMPERE_16",
    min_workers: int = 0,
    max_workers: int = 3
) -> Dict:
    """
    Create RunPod serverless endpoint
    
    Args:
        template_id: Template ID from create_template
        name: Endpoint name
        gpu_ids: GPU type (e.g., "AMPERE_16" for A4000/A5000)
        min_workers: Minimum number of workers (0 for auto-scale)
        max_workers: Maximum number of workers
    """
    print(f"\n{'='*60}")
    print(f"Creating endpoint: {name}")
    print(f"{'='*60}")
    
    runpod.api_key = RUNPOD_API_KEY
    
    try:
        # Create endpoint using RunPod SDK
        endpoint = runpod.create_endpoint(
            name=name,
            template_id=template_id,
            gpu_ids=gpu_ids,
            workers_min=min_workers,
            workers_max=max_workers,
            idle_timeout=60,  # Scale down after 60s of inactivity
        )
        
        print("✓ Endpoint created successfully!")
        print(f"  Endpoint ID: {endpoint['id']}")
        print(f"  GPU Type: {gpu_ids}")
        print(f"  Workers: {min_workers}-{max_workers}")
        print("  Idle Timeout: 60s")
        
        return endpoint
        
    except Exception as e:
        print(f"❌ Error creating endpoint: {e}")
        raise


def main():
    """Main deployment function"""
    print("\n" + "="*60)
    print("RunPod Serverless Deployment")
    print("="*60)
    
    if not RUNPOD_API_KEY:
        print("\n❌ Error: RUNPOD_API_KEY not set in environment")
        print("Please set it with: export RUNPOD_API_KEY=your-api-key")
        return
    
    # Create templates
    bi_template_id = create_template(BI_ENCODER_CONFIG)
    cross_template_id = create_template(CROSS_ENCODER_CONFIG)
    
    # Create endpoints
    bi_endpoint = create_endpoint(
        template_id=bi_template_id,
        name="legal-bi-encoder-endpoint",
        gpu_ids="AMPERE_16",
        min_workers=0,
        max_workers=3
    )
    
    cross_endpoint = create_endpoint(
        template_id=cross_template_id,
        name="legal-cross-encoder-endpoint",
        gpu_ids="AMPERE_16",
        min_workers=0,
        max_workers=3
    )
    
    # Summary
    print("\n" + "="*60)
    print("✓ Deployment Complete!")
    print("="*60)
    print(f"\nBi-Encoder Endpoint: {bi_endpoint['id']}")
    print(f"Cross-Encoder Endpoint: {cross_endpoint['id']}")
    print("\nAdd these to your .env file:")
    print(f"BI_ENCODER_ENDPOINT=https://api.runpod.ai/v2/{bi_endpoint['id']}")
    print(f"CROSS_ENCODER_ENDPOINT=https://api.runpod.ai/v2/{cross_endpoint['id']}")
    print("")


if __name__ == "__main__":
    main()