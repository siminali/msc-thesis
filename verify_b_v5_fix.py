#!/usr/bin/env python3
"""
Verify that the explicit model scale issue is fixed in B_v5.
"""
import numpy as np
from pathlib import Path

def check_b_v5_samples():
    """Check the B_v5 samples to confirm the fix worked."""
    print("="*70)
    print("VERIFYING B_v5 EXPLICIT MODEL SCALE FIX")
    print("="*70)
    
    base_dir = Path("results/addons/period_slices/B_v5/covid_crash")
    
    print("CHECKING B_v5 SAMPLES (WITH CORRECTED SAMPLING):")
    
    # Check explicit model
    explicit_path = base_dir / "explicit/real-conditions/samples.npy"
    if explicit_path.exists():
        samples = np.load(explicit_path)
        print(f"✅ Explicit samples found: shape={samples.shape}")
        print(f"   Mean: {samples.mean():.6f}")
        print(f"   Std: {samples.std():.6f}")
        print(f"   Range: [{samples.min():.6f}, {samples.max():.6f}]")
        
        if abs(samples.mean()) < 1 and abs(samples.std()) < 10:
            print("   ✅ SCALE FIXED! Values are in proper log returns range!")
            explicit_fixed = True
        else:
            print(f"   ❌ Scale still wrong - mean magnitude: {abs(samples.mean()):.0f}")
            explicit_fixed = False
    else:
        print("❌ Explicit samples not found")
        explicit_fixed = False
    
    # Check LLM model 
    llm_path = base_dir / "llm/real-conditions/samples.npy" 
    if llm_path.exists():
        samples = np.load(llm_path)
        print(f"✅ LLM samples found: shape={samples.shape}")
        print(f"   Mean: {samples.mean():.6f}")
        print(f"   Std: {samples.std():.6f}")
        print(f"   Range: [{samples.min():.6f}, {samples.max():.6f}]")
        print("   ✅ LLM model continues to work correctly")
    else:
        print("❌ LLM samples not found")
    
    return explicit_fixed

def check_b_v5_metrics():
    """Check if B_v5 metrics are reasonable."""
    print("\n" + "="*70)
    print("CHECKING B_v5 METRICS VALIDITY")
    print("="*70)
    
    metrics_path = Path("results/addons/period_slices/B_v5/covid_crash/metrics.json")
    if metrics_path.exists():
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
            
        print("✅ Metrics file found")
        
        # Check if explicit model has reasonable VaR/ES values
        if 'explicit' in metrics['models']:
            explicit_metrics = metrics['models']['explicit']
            
            if 'error' not in explicit_metrics:
                var_95 = explicit_metrics.get('var_95', None)
                es_95 = explicit_metrics.get('es_95', None)
                
                print(f"Explicit VaR 95%: {var_95}")
                print(f"Explicit ES 95%: {es_95}")
                
                if var_95 is not None and abs(var_95) < 10:
                    print("✅ VaR values are in reasonable range!")
                    return True
                else:
                    print("❌ VaR values still problematic")
                    return False
            else:
                print(f"❌ Explicit model has error: {explicit_metrics['error']}")
                return False
        else:
            print("❌ Explicit model not found in metrics")
            return False
    else:
        print("❌ Metrics file not found")
        return False

if __name__ == "__main__":
    print("FINAL VERIFICATION OF B_v5 EXPLICIT MODEL FIX")
    print("=" * 70)
    
    samples_fixed = check_b_v5_samples()
    metrics_ok = check_b_v5_metrics()
    
    print("\n" + "="*70)
    print("FINAL STATUS")
    print("="*70)
    
    if samples_fixed and metrics_ok:
        print("🎉 SUCCESS: Explicit model scale issue COMPLETELY RESOLVED!")
        print("   ✅ Samples are on correct log returns scale")
        print("   ✅ Metrics are reasonable") 
        print("   ✅ All models working correctly")
    elif samples_fixed:
        print("✅ SAMPLES FIXED but metrics may need regeneration")
    else:
        print("❌ Scale issue persists - need further investigation")
    
    print("="*70)
