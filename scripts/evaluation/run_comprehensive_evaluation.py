#!/usr/bin/env python3
"""
Simple Runner for Comprehensive Evaluation Pipeline
Runs the full evaluation pipeline for all three DDPM models
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from comprehensive_evaluation_pipeline import main

if __name__ == "__main__":
    main()
