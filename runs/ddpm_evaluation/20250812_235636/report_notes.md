# DDPM Model Evaluation Report Notes

## Train/Validation/Test Split
- **Training Period**: 90% of training data (chronological, no shuffling)
- **Validation Period**: 10% of training data (chronological, no shuffling)
- **Test Period**: 20% of full dataset (chronological, no shuffling)

## Observables
- **Returns**: Log returns computed from S&P 500 closing prices
- **Sequences**: Rolling windows of returns for temporal modeling
- **Noise**: Gaussian noise added during forward diffusion process

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

### Hill Tail Index
Estimates the tail index of heavy-tailed distributions. Lower values indicate heavier tails.

### Value at Risk (VaR)
The maximum expected loss at a given confidence level. 5% VaR means 95% of the time, losses won't exceed this threshold.

### Expected Shortfall (ES)
The average loss when VaR is exceeded, providing a measure of tail risk beyond VaR.

## Important Notes
- All evaluation metrics are computed using ONLY the test set
- Model is trained on training data only to prevent data leakage
- Validation set is used for early stopping during training
- Samples are generated to match the test set size
- Standardization parameters are computed from training data only
