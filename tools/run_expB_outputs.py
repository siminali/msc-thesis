#!/usr/bin/env python3
"""
CLI driver for generating all Experiment B (Controllability) outputs.

This script generates:
1. explicit_target_vs_realised_sigma_scatter.(pdf|png)
2. explicit_sigma_reliability_curve.(pdf|png)
3. regime_wise_performance.{tex,csv}
4. llm_news_bucket_distribution_comparison.(pdf|png)
5. llm_probe_diagnostics.{tex,csv}
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path to import our modules
sys.path.append(str(Path(__file__).parent))

from expB_plots import (
    plot_target_vs_realized_scatter,
    plot_reliability_curve,
    plot_llm_news_bucket_distribution
)
from expB_tables import (
    generate_regime_performance_table,
    generate_llm_probe_table
)

def main():
    parser = argparse.ArgumentParser(
        description="Generate Experiment B (Controllability) figures and tables"
    )
    parser.add_argument(
        "--expdir", 
        required=True,
        help="Path to experiment directory (e.g., results/addons/period_slices/B_v10)"
    )
    
    args = parser.parse_args()
    expdir = Path(args.expdir)
    
    # Validate directory exists
    if not expdir.exists():
        print(f"Error: Directory does not exist: {expdir}")
        sys.exit(1)
    
    print(f"Generating Experiment B outputs for: {expdir}")
    print("=" * 60)
    
    # Track generated outputs
    generated_outputs = []
    
    try:
        # 1. Generate explicit model target vs realized scatter plot
        print("\n1. Generating explicit target vs realised sigma scatter plot...")
        plot_target_vs_realized_scatter(str(expdir), "explicit")
        generated_outputs.extend([
            expdir / "explicit/figures/explicit_target_vs_realised_sigma_scatter.pdf",
            expdir / "explicit/figures/explicit_target_vs_realised_sigma_scatter.png"
        ])
        
        # 2. Generate explicit model reliability curve  
        print("\n2. Generating explicit sigma reliability curve...")
        plot_reliability_curve(str(expdir), "explicit")
        generated_outputs.extend([
            expdir / "explicit/figures/explicit_sigma_reliability_curve.pdf",
            expdir / "explicit/figures/explicit_sigma_reliability_curve.png"
        ])
        
        # 3. Generate regime-wise performance table
        print("\n3. Generating regime-wise performance table...")
        generate_regime_performance_table(str(expdir), "explicit")
        generated_outputs.extend([
            expdir / "explicit/tables/regime_wise_performance.tex",
            expdir / "explicit/tables/regime_wise_performance.csv"
        ])
        
        # 4. Generate LLM news bucket distribution comparison
        print("\n4. Generating LLM news bucket distribution comparison...")
        plot_llm_news_bucket_distribution(str(expdir), "llm")
        generated_outputs.extend([
            expdir / "llm/figures/llm_news_bucket_distribution_comparison.pdf",
            expdir / "llm/figures/llm_news_bucket_distribution_comparison.png"
        ])
        
        # 5. Generate LLM probe diagnostics table
        print("\n5. Generating LLM probe diagnostics table...")
        generate_llm_probe_table(str(expdir), "llm")
        generated_outputs.extend([
            expdir / "llm/tables/llm_probe_diagnostics.tex",
            expdir / "llm/tables/llm_probe_diagnostics.csv"
        ])
        
        print("\n" + "=" * 60)
        print("SUCCESS: All Experiment B outputs generated!")
        print("\nGenerated outputs:")
        
        for output in generated_outputs:
            if output.exists():
                print(f"✓ {output}")
            else:
                print(f"✗ {output} (not found)")
        
        print(f"\nTotal outputs: {len([o for o in generated_outputs if o.exists()])}/{len(generated_outputs)}")
        
    except Exception as e:
        print(f"\nERROR: Failed to generate outputs: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

