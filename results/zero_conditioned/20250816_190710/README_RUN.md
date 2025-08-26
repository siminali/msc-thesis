# Zero Conditioned Training Run Summary

## Run Information
- **Run ID**: 20250816_190710
- **Model**: Zero Conditioned
- **Timestamp**: 2025-08-16 19:29:01
- **Device**: cpu

## Key Parameters
- **Sequence Length**: 60
- **Volatility Window**: 20
- **Epochs**: 100
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Patience**: 10
- **Seed**: 42

## Training Results
- **Best Validation Loss**: 0.003605
- **Best Epoch**: N/A

## Generated Files
- `checkpoints/` - Model checkpoints
- `figures/` - Training curves and visualizations
- `tables/` - Evaluation tables
- `training_history.csv` - Training progress data
- `metadata.json` - Complete run metadata

## Next Steps
Run the evaluation pipeline using:
```bash
python src/evaluate_model.py --model-name zero_conditioned --run-id 20250816_190710
```
