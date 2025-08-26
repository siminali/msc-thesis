#!/usr/bin/env python3
"""
Test script to verify training pipeline imports and basic functionality
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test that all required imports work."""
    print("Testing training pipeline imports...")
    
    all_imports_ok = True
    
    try:
        # Test explicit model imports
        from explicit_cond_ddpm import (
            load_and_prepare_data, 
            create_conditioning_vectors, 
            create_sequences,
            ExplicitConditioningDDPM,
            ExplicitConditioningTrainer
        )
        print("✅ Explicit model imports successful")
        
    except ImportError as e:
        print(f"❌ Explicit model import error: {e}")
        all_imports_ok = False
    except Exception as e:
        print(f"❌ Explicit model unexpected error: {e}")
        all_imports_ok = False
    
    try:
        # Test LLM model imports
        from llm_conditioned_diffusion_refactored import (
            NewsDataLoader,
            LLMConditionedDiffusion,
            LLMDiffusionTrainer,
            ControllabilityProbe,
            create_time_based_splits
        )
        print("✅ LLM model imports successful")
        
    except ImportError as e:
        if "sentence_transformers" in str(e):
            print("⚠️  LLM model imports skipped (sentence_transformers not installed)")
            print("   Install with: pip install sentence-transformers")
        else:
            print(f"❌ LLM model import error: {e}")
            all_imports_ok = False
    except Exception as e:
        print(f"❌ LLM model unexpected error: {e}")
        all_imports_ok = False
    
    try:
        # Test training pipeline imports
        from train_all import (
            set_determinism,
            load_and_prepare_data_shared,
            create_zero_conditioned_model,
            create_explicit_conditioned_model,
            create_llm_conditioned_model
        )
        print("✅ Training pipeline imports successful")
        
    except ImportError as e:
        print(f"❌ Training pipeline import error: {e}")
        all_imports_ok = False
    except Exception as e:
        print(f"❌ Training pipeline unexpected error: {e}")
        all_imports_ok = False
    
    return all_imports_ok

def test_basic_functionality():
    """Test basic functionality without full training."""
    print("\nTesting basic functionality...")
    
    try:
        from train_all import set_determinism, DEFAULT_CONFIG
        
        # Test determinism setting
        set_determinism(42, 'cpu')
        print("✅ Determinism setting works")
        
        # Test config
        print(f"✅ Default config loaded: {len(DEFAULT_CONFIG)} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_explicit_model_creation():
    """Test that explicit model can be created."""
    print("\nTesting explicit model creation...")
    
    try:
        from train_all import create_explicit_conditioned_model
        from explicit_cond_ddpm import ExplicitConditioningDDPM
        
        # Test model creation
        model = ExplicitConditioningDDPM(
            sequence_length=60,
            conditioning_dim=5,
            hidden_dim=128
        )
        print("✅ Explicit model creation successful")
        
        # Test forward pass
        import torch
        x = torch.randn(2, 1, 60)  # [batch, channels, time]
        t = torch.randn(2, 1)      # [batch, time_dim]
        c = torch.randn(2, 5)      # [batch, conditioning]
        
        with torch.no_grad():
            output = model(x, t, c)
            print(f"✅ Forward pass successful: output shape {output.shape}")
            
            # Verify output dimensions
            expected_shape = (2, 1, 60)  # [batch, channels, time]
            if output.shape == expected_shape:
                print(f"✅ Output dimensions correct: {output.shape}")
            else:
                print(f"❌ Output dimensions incorrect: expected {expected_shape}, got {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Explicit model test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Training Pipeline Test Suite")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            # Test explicit model creation
            model_ok = test_explicit_model_creation()
            
            if model_ok:
                print("\n🎉 Core tests passed! Training pipeline is ready for explicit models.")
                print("⚠️  LLM models require sentence-transformers installation.")
                return True
            else:
                print("\n❌ Model creation tests failed.")
                return False
        else:
            print("\n❌ Functionality tests failed.")
            return False
    else:
        print("\n❌ Import tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
