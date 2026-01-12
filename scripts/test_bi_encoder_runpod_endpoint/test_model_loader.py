# test_model_loader.py
import os
from src.common.configuration import ConfigurationManager
from src.serve.bi_encoder_service.model_loader import BiEncoderModelLoader

def test_model_loader():
    """Test model loading from S3"""
    print("Testing model loader...")
    
    # Set AWS credentials
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        print("⚠️  AWS credentials not set. Set them first:")
        print("   export AWS_ACCESS_KEY_ID=xxx")
        print("   export AWS_SECRET_ACCESS_KEY=xxx")
        return False
    
    try:
        # Load config
        config_manager = ConfigurationManager()
        config = config_manager.get_bi_encoder_serve_config()
        
        print(f"Loading model: {config.model_id}")
        
        # Load model
        loader = BiEncoderModelLoader(config=config)
        model, tokenizer = loader.load_model()
        
        print("✅ Model loaded successfully")
        print(f"   Model type: {type(model)}")
        print(f"   Tokenizer: {type(tokenizer)}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_loader()