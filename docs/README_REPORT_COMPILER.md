# Report Compiler: Professional PDF Generation

Automated PDF report generation from experiment results with comprehensive analysis, professional layout, and robust error handling.

## Overview

The Report Compiler transforms experiment outputs into publication-ready PDF reports that include:
- **Performance Overview Tables**: Model metrics with VaR, ES, and volatility
- **Key Figures**: ECDF overlays, Q-Q plots, VaR/ES analysis, volatility tracking
- **Findings Synthesis**: Automated insights from metrics and findings files
- **Professional Layout**: Consistent typography, styling, and formatting
- **Robust Error Handling**: Red "SKIPPED" boxes for missing assets, never fails

## Quick Start

### Basic Usage

```bash
# Generate report for Experiment A
python report_compiler.py --expdir results/addons/period_slices/A

# Generate report for Experiment B
python report_compiler.py --expdir results/addons/period_slices/B

# Use versioned experiment directories
python report_compiler.py --expdir results/addons/period_slices/A_v8
python report_compiler.py --expdir results/addons/period_slices/B_v3
```

### Output Location

Reports are automatically saved as:
```
<experiment_directory>/report_<experiment_name>.pdf
```

Examples:
- `results/addons/period_slices/A/report_A.pdf`
- `results/addons/period_slices/B/report_B.pdf`
- `results/addons/period_slices/A_v8/report_A_v8.pdf`

## Features

### 📊 **Performance Overview Tables**
Automatically extracted from `metrics.json`:
- Model names and status
- VaR (5%) and ES (5%) values
- Volatility (standard deviation)
- Success/failure indicators

### 📈 **Key Figures Integration**
Includes publication-quality plots:
- **ECDF Overlay**: Empirical distribution comparisons
- **Q-Q Plots**: Quantile-quantile analysis for both tails
- **VaR/ES Analysis**: Risk metric visualization with exceedance timeline
- **Volatility Tracking**: Realized volatility with RMSE performance

### 🔍 **Findings Synthesis**
Automated extraction from `findings.jsonl`:
- Model performance summaries
- Statistical significance tests (Diebold-Mariano)
- Controllability insights (Experiment B)
- Risk metric comparisons

### 🛡️ **Robust Error Handling**
- **Never Fails**: Continues compilation even with missing assets
- **Red SKIPPED Boxes**: Clear indication of missing components
- **Graceful Degradation**: Partial reports with available content
- **Detailed Logging**: Clear error messages and warnings

### 🎨 **Professional Layout**
- **Consistent Typography**: Professional fonts and sizing
- **Section Organization**: Clear hierarchy with headers
- **Table Formatting**: Clean, readable data presentation
- **Figure Captions**: Descriptive captions for all plots

## Report Structure

### Title Page
- Experiment name and description
- Report generation metadata
- Windows summary table
- Compilation statistics

### Window Sections (per stress/test period)
For each window (e.g., `covid_crash`):

#### Performance Overview
Table showing:
```
Model     | VaR (5%) | ES (5%) | Volatility | Status
----------|----------|---------|------------|--------
explicit  | -5.392   | -8.260  | 4.862      | Success
zero      | -14.851  | -17.038 | 9.887      | Success
llm       | -12.705  | -15.703 | 5.023      | Success
```

#### Key Figures
Four professional plots:
1. **ECDF Overlay**: Model vs real data distributions
2. **Q-Q Plots**: Tail behavior analysis
3. **VaR/ES Analysis**: Risk metrics with exceedance tracking
4. **Volatility Tracking**: Realized volatility comparison

#### Key Findings
Synthesized insights:
```
Model Performance:
• explicit: VaR(5%) = -5.392
• zero: VaR(5%) = -14.851

Model Comparisons:
• explicit_vs_zero: significant difference (p=0.0011)

Controllability:
• llm model is 46.0% more conservative
```

## Dependencies

### Required Packages
```bash
pip install reportlab
```

### File Requirements
The compiler expects this structure:
```
<experiment_dir>/
├── findings.jsonl                    # Optional: findings data
└── <window_id>/
    ├── metrics.json                  # Required: performance metrics
    └── figs/                        # Required: figure directory
        ├── ecdf_overlay.png
        ├── qq_plots.png
        ├── var_es_analysis.png
        └── realized_volatility.png
```

## Automatic Asset Discovery

### Flexible Path Resolution
The compiler automatically finds assets in multiple locations:

1. **Versioned Directories**: `A_v8/covid_crash/metrics.json`
2. **Base Directories**: `A/covid_crash/metrics.json` (fallback)
3. **Figure Locations**: Checks both versioned and base directories

### Missing Asset Handling
When assets are missing:
- **Metrics**: Shows "SKIPPED" section with error message
- **Figures**: Displays red "SKIPPED" box with asset name
- **Findings**: Shows "No findings available" message
- **Compilation**: Continues successfully with partial content

## Example Output

### Successful Compilation
```bash
$ python report_compiler.py --expdir results/addons/period_slices/A_v8

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

### Error Handling Example
```bash
2025-08-30 19:22:12,888 - INFO - Found metrics in base directory: results/addons/period_slices/A/covid_crash/metrics.json
2025-08-30 19:22:12,888 - INFO - Loaded metrics for window: covid_crash
```

## Advanced Usage

### Command Line Arguments

| Argument | Description | Required | Example |
|----------|-------------|----------|---------|
| `--expdir` | Experiment directory path | Yes | `results/addons/period_slices/A` |

### Supported Experiment Types

#### Experiment A (Stress Testing)
- **Focus**: Out-of-sample stress testing
- **Models**: Zero, Explicit, LLM
- **Windows**: Stress periods (COVID crash, recovery)
- **Findings**: Model comparisons, risk metrics

#### Experiment B (Controllability)
- **Focus**: Counterfactual controllability testing
- **Models**: Explicit, LLM (controllable models)
- **Modes**: Real-conditions, Calm-conditions, LLM-knob
- **Findings**: Controllability insights, condition effects

### Multi-Window Support
Automatically processes all windows in experiment directory:
```bash
# Processes all windows found in experiment
python report_compiler.py --expdir results/addons/period_slices/A

# Example windows: covid_crash, covid_recovery, post_covid
```

## Customization

### Adding New Figure Types
To include additional plots, modify the `compile_window_section` method:

```python
# Add new figure
new_fig_path = figs_dir / 'new_analysis.png'
self.add_figure_or_skipped(elements, new_fig_path, f"New Analysis - {window_id}")
```

### Custom Styles
Modify `_setup_custom_styles()` to adjust:
- Font sizes and colors
- Section spacing
- Table formatting
- Text alignment

### Extended Findings
Enhance `create_findings_paragraph()` to include:
- Additional metrics
- Custom analysis
- Model-specific insights

## Troubleshooting

### Common Issues

#### Missing Metrics Files
```
WARNING - Metrics file not found: results/addons/period_slices/A_v8/covid_crash/metrics.json
INFO - Found metrics in base directory: results/addons/period_slices/A/covid_crash/metrics.json
```
**Solution**: Compiler automatically checks base directories. No action needed.

#### Missing Figure Files
Red "SKIPPED" boxes appear in PDF.
**Solution**: Ensure plotting pipeline has run successfully. Check:
```bash
ls results/addons/period_slices/A/covid_crash/figs/
```

#### PDF Size Issues
```
Report Size: 0.00 MB  # Too small - missing content
Report Size: 10+ MB   # Too large - check image sizes
```
**Solution**: 
- Small size: Check metrics and figures availability
- Large size: Consider reducing image quality or size

#### Import Errors
```
ModuleNotFoundError: No module named 'reportlab'
```
**Solution**: Install dependencies:
```bash
pip install reportlab
```

### Debug Commands

```bash
# Check experiment structure
find results/addons/period_slices/A -name "*.json" -o -name "*.png"

# Verify findings content
cat results/addons/period_slices/A_v8/findings.jsonl | jq .

# Check figure availability
ls -la results/addons/period_slices/A/covid_crash/figs/
```

## Integration with Pipeline

### Manual Report Generation
```bash
# After running integrated pipeline
python experiment_A_evaluator_v2.py --window covid_crash --num-paths 1000
python report_compiler.py --expdir results/addons/period_slices/A_v8
```

### Batch Report Generation
```bash
#!/bin/bash
# Generate reports for all experiments
for exp in A B A_v8 B_v3; do
    if [ -d "results/addons/period_slices/$exp" ]; then
        echo "Generating report for $exp..."
        python report_compiler.py --expdir "results/addons/period_slices/$exp"
    fi
done
```

## Performance Characteristics

### Compilation Time
- **Single Window**: ~2-5 seconds
- **Multiple Windows**: ~5-15 seconds total
- **Large Images**: May increase to ~10-30 seconds

### File Sizes
- **Typical Report**: 0.5-2 MB
- **With High-Res Images**: 1-5 MB
- **Multiple Windows**: 2-10 MB

### Memory Usage
- **Peak Usage**: ~50-200 MB during compilation
- **Steady State**: ~20-50 MB
- **Large Reports**: May require up to 500 MB

## Best Practices

### 1. **Complete Pipeline First**
Run the full integrated pipeline before generating reports:
```bash
python experiment_A_evaluator_v2.py --window covid_crash --num-paths 1000
python report_compiler.py --expdir results/addons/period_slices/A_v8
```

### 2. **Check Asset Availability**
Verify all required files exist:
```bash
# Check metrics
find results/addons/period_slices -name "metrics.json"

# Check figures
find results/addons/period_slices -name "figs" -type d
```

### 3. **Use Base Directories for Final Reports**
For final publication, use base experiment directories:
```bash
python report_compiler.py --expdir results/addons/period_slices/A
python report_compiler.py --expdir results/addons/period_slices/B
```

### 4. **Archive Generated Reports**
```bash
# Create report archive
mkdir reports_$(date +%Y%m%d)
cp results/addons/period_slices/*/report_*.pdf reports_$(date +%Y%m%d)/
```

## Future Enhancements

### Potential Extensions
- **Multi-Experiment Comparison**: Cross-experiment analysis reports
- **Interactive Elements**: Clickable references and navigation
- **LaTeX Integration**: Academic paper formatting
- **Custom Templates**: User-defined report layouts
- **Automated Insights**: AI-generated findings summaries
- **Web Reports**: HTML output with interactive plots

### Technical Improvements
- **PDF Optimization**: Smaller file sizes with vector graphics
- **Parallel Processing**: Faster compilation for large experiments
- **Template System**: Configurable report layouts
- **Real-time Updates**: Live report generation during experiments

The Report Compiler provides a professional, automated solution for generating publication-ready analysis reports from experimental results, ensuring consistent formatting and comprehensive coverage of all key findings and visualizations.
