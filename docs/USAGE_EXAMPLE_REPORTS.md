# Usage Examples: Report Compiler

Complete examples demonstrating the report compiler for different experimental scenarios.

## Quick Start Examples

### Experiment A: Stress Testing Report
```bash
# Generate comprehensive stress testing report
python report_compiler.py --expdir results/addons/period_slices/A_v8

# Output: results/addons/period_slices/A_v8/report_A_v8.pdf
```

### Experiment B: Controllability Report  
```bash
# Generate controllability testing report
python report_compiler.py --expdir results/addons/period_slices/B_v3

# Output: results/addons/period_slices/B_v3/report_B_v3.pdf
```

### Base Experiment Directories
```bash
# Use base directories for final publication reports
python report_compiler.py --expdir results/addons/period_slices/A
python report_compiler.py --expdir results/addons/period_slices/B
```

## Expected Output

### Successful Compilation
```
============================================================
REPORT COMPILATION SUMMARY
============================================================
Experiment: A_v8
Source Directory: results/addons/period_slices/A_v8
Output PDF: results/addons/period_slices/A_v8/report_A_v8.pdf
Windows Processed: 1
Findings Included: 1
Report Size: 0.86 MB

Report compiled successfully! 📄✨
```

### Key Features Demonstrated

#### Automatic Asset Discovery
```
INFO - Found metrics in base directory: results/addons/period_slices/A/covid_crash/metrics.json
INFO - Loaded metrics for window: covid_crash
```

#### Professional PDF Generation
- **File Size**: ~0.86-0.91 MB (includes high-quality figures)
- **Content**: Complete analysis with tables, figures, and findings
- **Layout**: Professional typography and consistent formatting

## Report Contents

### Experiment A Report Structure
1. **Title Page**
   - "EXPERIMENT A REPORT"
   - "Out-of-Sample Stress Testing"
   - Report metadata and windows summary

2. **COVID Crash Window Section**
   - **Performance Overview Table**:
     ```
     Model     | VaR (5%) | ES (5%) | Volatility | Status
     ----------|----------|---------|------------|--------
     explicit  | -5.392   | -8.260  | 4.862      | Success
     zero      | -14.851  | -17.038 | 9.887      | Success
     ```
   
   - **Key Figures** (PNG format):
     - ECDF Overlay comparison
     - Q-Q plots for tail analysis
     - VaR/ES analysis with exceedances  
     - Realized volatility tracking
   
   - **Key Findings**:
     ```
     Model Performance:
     • explicit: VaR(5%) = -5.392
     • zero: VaR(5%) = -14.851
     
     Model Comparisons:
     • explicit_vs_zero: significant difference (p=0.0011)
     ```

### Experiment B Report Structure
1. **Title Page**
   - "EXPERIMENT B REPORT"
   - "Counterfactual Controllability Testing"
   - Report metadata and windows summary

2. **COVID Crash Window Section**
   - **Performance Overview Table**:
     ```
     Model     | VaR (5%) | ES (5%) | Volatility | Status
     ----------|----------|---------|------------|--------
     llm       | -12.705  | -15.703 | 5.023      | Success
     explicit  | -8.703   | -11.521 | 5.426      | Success
     ```
   
   - **Key Figures**: Same 4 plots as Experiment A
   
   - **Key Findings**:
     ```
     Model Performance:
     • llm: VaR(5%) = -12.705
     • explicit: VaR(5%) = -8.703
     
     Model Comparisons:
     • llm_vs_explicit: not significant difference (p=0.8282)
     
     Controllability:
     • llm model is 46.0% more conservative
     ```

## Integration Workflow

### Complete End-to-End Example
```bash
# 1. Run integrated experiment pipeline
python experiment_A_evaluator_v2.py --window covid_crash --num-paths 1000 --csv-file sp500_data.csv

# 2. Generate professional report
python report_compiler.py --expdir results/addons/period_slices/A_v8

# 3. View results
ls -lh results/addons/period_slices/A_v8/report_A_v8.pdf
# Output: 898KB PDF with complete analysis
```

### Batch Report Generation
```bash
#!/bin/bash
# Generate reports for all available experiments

echo "Generating experiment reports..."

for expdir in results/addons/period_slices/A*; do
    if [ -d "$expdir" ]; then
        echo "Processing: $expdir"
        python report_compiler.py --expdir "$expdir"
    fi
done

for expdir in results/addons/period_slices/B*; do
    if [ -d "$expdir" ]; then
        echo "Processing: $expdir"
        python report_compiler.py --expdir "$expdir"
    fi
done

echo "Reports generated:"
find results/addons/period_slices -name "report_*.pdf" -exec ls -lh {} \;
```

## Asset Requirements

### Required Directory Structure
```
<experiment_dir>/
├── findings.jsonl              # Findings data (optional)
└── <window_id>/
    ├── metrics.json            # Performance metrics (required)
    └── figs/                   # Figures directory (required)
        ├── ecdf_overlay.png
        ├── qq_plots.png  
        ├── var_es_analysis.png
        └── realized_volatility.png
```

### Cross-Directory Asset Discovery
The compiler automatically searches:
1. **Versioned directory**: `A_v8/covid_crash/metrics.json`
2. **Base directory**: `A/covid_crash/metrics.json` (fallback)
3. **Figure locations**: Both versioned and base directories

## Error Handling Examples

### Missing Assets Handling
```bash
# Missing figures example
python report_compiler.py --expdir results/addons/period_slices/missing_test

# Result: PDF with red "SKIPPED" boxes where figures should be
# Report still compiles successfully with available content
```

### Graceful Degradation
- **Missing metrics.json**: Shows "SKIPPED" section 
- **Missing figures**: Red "SKIPPED" boxes with asset names
- **Missing findings.jsonl**: "No findings available" message
- **Empty directory**: Basic structure with placeholder content

## Real Performance Data

### File Sizes (from actual runs)
```bash
$ find results/addons/period_slices -name "report_*.pdf" -exec ls -lh {} \;
-rw-r--r-- 1 user staff 878K Aug 30 19:22 results/addons/period_slices/A/report_A.pdf
-rw-r--r-- 1 user staff 878K Aug 30 19:22 results/addons/period_slices/A_v8/report_A_v8.pdf  
-rw-r--r-- 1 user staff 889K Aug 30 19:22 results/addons/period_slices/B_v3/report_B_v3.pdf
```

### Compilation Performance
- **Single Window**: 2-3 seconds
- **Multiple Windows**: Scales linearly  
- **Memory Usage**: ~50-200 MB peak
- **File Quality**: High-resolution figures with vector text

## Professional Features

### Typography and Layout
- **Consistent fonts**: Professional Helvetica family
- **Color scheme**: Blue headers, black text, red warnings
- **Table formatting**: Clean borders and alternating backgrounds
- **Figure captions**: Descriptive and properly positioned

### Content Organization
- **Hierarchical structure**: Clear section and subsection headers
- **Logical flow**: Overview → Figures → Findings
- **Page breaks**: Clean separation between windows
- **Professional spacing**: Appropriate margins and padding

### Publication Quality
- **Vector text**: Crisp text at any zoom level
- **High-res images**: PNG figures at publication quality
- **Consistent styling**: Professional appearance throughout
- **Comprehensive content**: All analysis components included

## Troubleshooting

### Common Issues and Solutions

#### Asset Location Problems
```bash
# Check asset availability
find results/addons/period_slices/A -name "metrics.json"
find results/addons/period_slices/A -name "*.png" | head -5
```

#### PDF Generation Issues
```bash
# Verify ReportLab installation
python -c "import reportlab; print('ReportLab OK')"

# Check file permissions
ls -la results/addons/period_slices/A_v8/
```

#### Content Missing
```bash
# Verify experiment completion
cat results/addons/period_slices/A_v8/findings.jsonl | jq .
ls -la results/addons/period_slices/A/covid_crash/figs/
```

The report compiler provides a robust, professional solution for generating publication-ready PDF reports from experimental results, with comprehensive error handling and automatic asset discovery.
