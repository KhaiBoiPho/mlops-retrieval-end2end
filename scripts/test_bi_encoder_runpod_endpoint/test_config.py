# test_config.py
from src.common.configuration import ConfigurationManager

def test_config():
    """Test config loading"""
    print("Testing config loading...")
    
    try:
        config_manager = ConfigurationManager()
        config = config_manager.get_bi_encoder_serve_config()
        
        print("✅ Config loaded successfully")
        print(f"   S3 Bucket: {config.s3_bucket}")
        print(f"   Model ID: {config.model_id}")
        print(f"   Device: {config.device}")
        print(f"   Batch Size: {config.batch_size}")
        print(f"   Max Seq Length: {config.max_seq_length}")
        
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

if __name__ == "__main__":
    test_config()