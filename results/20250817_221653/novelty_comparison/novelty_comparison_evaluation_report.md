# Novelty Models Comparison Report

## zero_conditioned
- KS=0.5310 (p=0.0000), W=0.8949, MMD=1.5260
- Forecast: MSE=1.410368, MAE=0.869898, RMSE=1.187589
- Risk: VaR95=-1.8028, ES95=-2.5964, Kupiec p=nan, CC p=nan

## explicit_conditioned
- KS=0.5887 (p=0.0000), W=2.1003, MMD=8.0395
- Forecast: MSE=8.445558, MAE=2.110221, RMSE=2.906124
- Risk: VaR95=-3.4603, ES95=-5.3444, Kupiec p=nan, CC p=nan

## llm_conditioned
- KS=0.5182 (p=0.0000), W=24.3212, MMD=976.5688
- Forecast: MSE=882.769744, MAE=23.116069, RMSE=29.711441
- Risk: VaR95=-47.4464, ES95=-63.9590, Kupiec p=nan, CC p=nan
