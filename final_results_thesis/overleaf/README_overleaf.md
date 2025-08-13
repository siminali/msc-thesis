# Overleaf Export Assets

This directory contains LaTeX tables and figure stubs for integration into Overleaf.

## Tables

- **tab:basic_stats**: Basic Statistics: Mean, standard deviation, skewness, kurtosis, and quartiles for Real and synthetic data.
- **tab:distribution_tests**: Distribution Tests: Kolmogorov-Smirnov (KS), Anderson-Darling (AD), and Maximum Mean Discrepancy (MMD) statistics.
- **tab:var_es**: Value at Risk (VaR) and Expected Shortfall (ES) at different confidence levels. Negative values indicate downside risk.
- **tab:backtesting**: VaR Backtesting Results: Observed violations, expected rates, and test p-values. Violation intervals use Clopper-Pearson method.
- **tab:temporal_tests**: Temporal Dependence Tests: Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) at key lags.
- **tab:volatility_metrics**: Volatility Metrics: Mean rolling 20-day volatility, ACF at lag 1, persistence proxy, and volatility-of-volatility.
- **tab:robustness_bootstrap**: Robustness Analysis: Bootstrap statistics with mean ± standard deviation and 95% confidence intervals across multiple runs.
- **tab:evt_tailindex**: Extreme Value Theory: Hill tail index estimates for the left tail with 95% confidence intervals.
- **tab:capital_impact**: Capital Impact: Expected Shortfall at 99% confidence level and differences versus Real data.
- **tab:ranking_table**: Overall Ranking: Component scores and final ranking across all evaluation dimensions.

## Figure Stubs

- **fig:distribution_comparison**: Distribution Comparison: Real versus synthetic return densities on shared axes.
- **fig:cdf_comparison**: Cumulative Distribution Function: Real versus synthetic return CDFs on shared axes.
- **fig:var_comparison**: Value at Risk Comparison: VaR at different confidence levels across models.
- **fig:acf_comparison**: Autocorrelation Function: Returns ACF up to lag 20 with 95% confidence bands.
- **fig:rolling_volatility**: Rolling Volatility: 20-day rolling volatility comparison across models.
- **fig:tail_analysis**: Tail Analysis: Left and right tail density zoom panels on identical ranges.
- **fig:condition_response**: Condition→Response Analysis: Targeted versus realized volatility with regression line and metrics.

## Usage

1. **Tables**: Use `\input{tables/filename.tex}` to include tables
2. **Figures**: Use `\input{figures/filename.tex}` to include figure stubs
3. **PDFs**: Figures are saved as vector PDFs in the figures/ directory

## Notes

- Check the notes/ directory for any warnings or missing data notes
- All values match the thesis methodology exactly
- No model retraining or data regeneration performed
