# test_embedding_local.py
import os
from src.common.configuration import ConfigurationManager
from src.serve.bi_encoder_service.model_loader import BiEncoderModelLoader
from src.serve.bi_encoder_service.embedding import BiEncoderEmbedder

def test_embedding():
    """Test embedding generation"""
    print("Testing embedding generation...")
    
    try:
        # Load config
        config_manager = ConfigurationManager()
        config = config_manager.get_bi_encoder_serve_config()
        
        # Load model
        loader = BiEncoderModelLoader(config=config)
        model, tokenizer = loader.load_model()
        
        # Initialize embedder
        embedder = BiEncoderEmbedder(
            model=model,
            tokenizer=tokenizer,
            config=config
        )
        
        # Test single embed
        print("\n1. Testing single embed...")
        text = "Luật dân sự về quyền sở hữu tài sản"
        embedding, encode_time = embedder.encode_single(text)
        
        print("   ✅ Single embed successful")
        print(f"   - Dimension: {len(embedding)}")
        print(f"   - Encode time: {encode_time:.2f}ms")
        print(f"   - Embedding preview: {embedding[:5]}")
        
        # Test batch embed
        print("\n2. Testing batch embed...")
        texts = [
            "Quy định về thuế thu nhập cá nhân",
            "Luật bảo vệ môi trường",
            "Hợp đồng lao động"
        ]
        embeddings, encode_time = embedder.encode_batch(texts)
        
        print("   ✅ Batch embed successful")
        print(f"   - Count: {len(embeddings)}")
        print(f"   - Total time: {encode_time:.2f}ms")
        print(f"   - Avg time: {encode_time/len(texts):.2f}ms/text")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set AWS credentials first
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        print("Please set AWS credentials:")
        print("export AWS_ACCESS_KEY_ID=xxx")
        print("export AWS_SECRET_ACCESS_KEY=xxx")
    else:
        test_embedding()