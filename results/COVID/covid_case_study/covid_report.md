# COVID-19 Crisis Case Study

Window: 2020-02-20 to 2020-04-30 (length=50 days)

This report compares real crisis returns to synthetic crisis-period paths from:
- Zero-conditioned (unconditional)
- Explicit-conditioned (regime + target volatility)
- LLM-conditioned (news embeddings)

Key outputs:
- crisis_returns_paths.pdf: Overlaid real vs synthetic crash returns
- crisis_var_es_curves.pdf: VaR/ES curves (95%/99% highlighted)
- crisis_exceedance_timelines.pdf: VaR exceedance timelines
- crisis_rolling_volatility.pdf: Rolling volatility overlay
- crisis_breaches.csv/json: Breaches and p-values (Kupiec, Christoffersen)

Observations:
- Unconditional provides a baseline shock replication without control.
- Explicit conditioning targets Down-High regimes, improving crash-like behavior.
- LLM conditioning captures news-driven dynamics; alignment depends on news embeddings.