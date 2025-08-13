# GARCH Model Evaluation Report Notes

## Train/Test Split
- **Training Period**: 80% of data (chronological, no shuffling)
- **Test Period**: 20% of data (chronological, no shuffling)
- **Validation**: None (GARCH uses MLE on training data only)

## Observables
- **Returns**: Log returns computed from S&P 500 closing prices
- **Volatility**: Conditional volatility from GARCH(1,1) model
- **VaR**: Value at Risk at 5% confidence level

## Rolling Volatility Definition
Rolling volatility is computed using a 20-day window standard deviation of returns, providing a smoothed view of volatility evolution over time.

## QQ Plot Interpretation
QQ plots compare the empirical distribution of returns against theoretical normal distribution. Points following the diagonal line indicate normality, while deviations suggest non-normal behavior.

## ACF Interpretation
Autocorrelation Function (ACF) measures the correlation between returns at different time lags. Significant correlations at lag 1 suggest momentum effects, while correlations in absolute returns indicate volatility clustering.

## Statistical Tests

### Kolmogorov-Smirnov (KS) Test
Tests whether two samples come from the same distribution. Low p-values indicate significant differences.

### Anderson-Darling (AD) Test
More sensitive to differences in the tails than KS test. Higher values indicate greater deviation from the reference distribution.

### Maximum Mean Discrepancy (MMD)
Measures the distance between two probability distributions using kernel methods. Lower values indicate more similar distributions.

### Value at Risk (VaR)
The maximum expected loss at a given confidence level. 5% VaR means 95% of the time, losses won't exceed this threshold.

### Expected Shortfall (ES)
The average loss when VaR is exceeded, providing a measure of tail risk beyond VaR.

### Kupiec Test
Tests whether the observed VaR violation rate matches the expected rate. Low p-values suggest VaR underestimation.

### Christoffersen Test
Tests whether VaR violations are independent over time. Low p-values suggest clustering of violations.

## Important Notes
- All evaluation metrics (VaR, ES, backtests) are computed using ONLY the test set
- Model is fitted on training data only to prevent data leakage
- Forecasts are generated for the test period only
- In-sample volatility plots show training period only
