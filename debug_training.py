#!/usr/bin/env python3
"""
Debug script to test training pipeline step by step
"""

import time
import sys
import os
sys.path.append('src')

def test_step(step_name, func, *args, **kwargs):
    """Test a single step with timing."""
    print(f"\n{'='*60}")
    print(f"TESTING: {step_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"✅ {step_name} completed in {elapsed:.2f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {step_name} failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("Debugging training pipeline step by step...")
    
    # Test 1: Basic imports
    test_step("Import training modules", lambda: __import__('train_all'))
    
    # Test 2: Data loading
    try:
        from train_all import load_and_prepare_data_shared, DEFAULT_CONFIG
        config = DEFAULT_CONFIG.copy()
        config['device'] = 'cpu'
        
        test_step("Data loading", load_and_prepare_data_shared, config)
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
    
    # Test 3: Zero model creation
    try:
        from train_all import create_zero_conditioned_model
        test_step("Zero model creation", create_zero_conditioned_model, config)
    except Exception as e:
        print(f"❌ Zero model creation failed: {e}")
    
    # Test 4: Explicit model creation
    try:
        from train_all import create_explicit_conditioned_model
        test_step("Explicit model creation", create_explicit_conditioned_model, config)
    except Exception as e:
        print(f"❌ Explicit model creation failed: {e}")
    
    # Test 5: LLM model creation
    try:
        from train_all import create_llm_conditioned_model
        test_step("LLM model creation", create_llm_conditioned_model, config)
    except Exception as e:
        print(f"❌ LLM model creation failed: {e}")
    
    print("\n" + "="*60)
    print("DEBUG COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
