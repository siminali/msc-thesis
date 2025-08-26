# 🚀 Comprehensive Training Pipeline for Three DDPM Models

## Overview

This training pipeline (`src/train_all.py`) provides a **single, reproducible training system** for all three DDPM conditioning approaches:

1. **Zero-Conditioned DDPM** - Unconditional baseline
2. **Explicit-Conditioned DDPM** - Regime + volatility conditioning  
3. **LLM-Conditioned DDPM** - News-based semantic conditioning

## ✨ Key Features

### **🔄 Reproducibility**
- **Full determinism**: Global seeds, CUDA deterministic flags, benchmark disabled
- **Identical data loading**: Same preprocessing, same splits for fair comparison
- **Consistent training**: Same hyperparameters, same validation strategy

### **📊 Standardized Outputs**
- **Organized results**: `results/<model_name>/<run_id>/`
- **Complete metadata**: Args, seeds, device, git hash, training history
- **Resume capability**: Continue training from existing checkpoints
- **Publication-ready**: Training curves, LaTeX tables, comprehensive logs

### **🔧 Flexible Configuration**
- **Shared config**: Single JSON file for all parameters
- **CLI overrides**: Override any parameter from command line
- **Model selection**: Train individual models or all at once
- **Auto-detection**: Automatic data path detection and device selection

## 🛠️ Installation

### Prerequisites
```bash
pip install -r requirements_llm_refactored.txt
```

### Verify Imports
```bash
python -c "from src.train_all import *; print('All imports successful')"
```

## 🚀 Quick Start

### **1. Train All Models (Default)**
```bash
python src/train_all.py
```

### **2. Train Specific Models**
```bash
# Train only zero-conditioned model
python src/train_all.py --models zero

# Train explicit and LLM models
python src/train_all.py --models explicit llm

# Train all models
python src/train_all.py --models all
```

### **3. Use Custom Configuration**
```bash
python src/train_all.py --config configs/training_config.json
```

### **4. Override Parameters**
```bash
python src/train_all.py --epochs 200 --batch-size 32 --lr 5e-4
```

## 📋 Configuration Options

### **Data Parameters**
- `seq_len`: Sequence length for training (default: 60)
- `vol_window`: Volatility rolling window (default: 20)
- `data_path`: Path to S&P 500 data (auto-detected if None)

### **Training Parameters**
- `epochs`: Number of training epochs (default: 100)
- `batch_size`: Training batch size (default: 64)
- `lr`: Learning rate (default: 1e-3)
- `patience`: Early stopping patience (default: 10)
- `grad_clip`: Gradient clipping value (default: 1.0)

### **Model Parameters**
- `num_timesteps`: Diffusion timesteps (default: 1000)
- `beta_schedule`: Beta schedule (cosine/linear, default: cosine)
- `sampler`: Sampling method (ddpm/ddim, default: ddim)
- `sample_steps`: Sampling steps (default: 50)

### **CFG Parameters**
- `cfg_p`: Conditioning dropout probability (default: 0.1)
- `cfg_scale`: Classifier-free guidance scale (default: 7.5)

### **LLM-Specific Parameters**
- `encoder_name`: Sentence transformer model (default: all-MiniLM-L6-v2)
- `target_embedding_dim`: Target embedding dimension after PCA (default: 64)
- `cache_dir`: News embeddings cache directory

### **System Parameters**
- `seed`: Random seed for reproducibility (default: 42)
- `device`: Device selection (auto/cpu/cuda, default: auto)
- `results_dir`: Results directory (default: results)

## 📁 Output Structure

### **Directory Layout**
```
results/
├── zero_conditioned/
│   └── 20241201_143022/
│       ├── checkpoints/
│       │   ├── best_model.pth
│       │   └── final_model.pth
│       ├── figures/
│       │   └── training_curves.pdf
│       ├── tables/
│       ├── training_history.csv
│       ├── training_history.json
│       ├── metadata.json
│       └── README_RUN.md
├── explicit_conditioned/
│   └── 20241201_143022/
│       ├── checkpoints/
│       ├── figures/
│       ├── tables/
│       ├── training_history.csv
│       ├── training_history.json
│       ├── metadata.json
│       ├── conditioning_metadata.json
│       └── README_RUN.md
└── llm_conditioned/
    └── 20241201_143022/
        ├── checkpoints/
        ├── figures/
        ├── tables/
        ├── training_history.csv
        ├── training_history.json
        ├── metadata.json
        ├── controllability_probe.pkl
        ├── probe_diagnostics.json
        └── README_RUN.md
```

### **Generated Files**

#### **Checkpoints**
- `best_model.pth`: Best model based on validation loss
- `final_model.pth`: Final model after training completion

#### **Training History**
- `training_history.csv`: Epoch-by-epoch training progress
- `training_history.json`: Complete training history with metadata

#### **Visualizations**
- `training_curves.pdf`: Training and validation loss curves
- `figures/`: Additional training visualizations

#### **Metadata**
- `metadata.json`: Complete run information (args, seed, device, git hash)
- `conditioning_metadata.json`: Explicit model conditioning parameters
- `probe_diagnostics.json`: LLM model controllability probe diagnostics

#### **Documentation**
- `README_RUN.md`: Run summary for immediate evaluation pipeline use

## 🔄 Resume Training

### **Automatic Resume**
The pipeline automatically detects existing checkpoints and resumes training:

```bash
# If checkpoint exists, training resumes automatically
python src/train_all.py --models explicit
```

### **Resume Behavior**
- **Loads best model**: Restores model weights and optimizer state
- **Continues from next epoch**: No duplicate training
- **Maintains progress**: Best validation loss and patience counter preserved

## 📊 Training Process

### **1. Zero-Conditioned Model**
- **Architecture**: Same as explicit model but with zero conditioning
- **Purpose**: Unconditional baseline for comparison
- **Training**: Standard DDPM training without CFG

### **2. Explicit-Conditioned Model**
- **Architecture**: Regime one-hot + volatility scalar conditioning
- **Purpose**: Interpretable, structured conditioning
- **Training**: With CFG dropout and guidance

### **3. LLM-Conditioned Model**
- **Architecture**: News embeddings with PCA reduction
- **Purpose**: Semantic, unstructured conditioning
- **Training**: With CFG dropout, guidance, and controllability probe

## 🎯 Advanced Usage

### **Custom Configuration File**
Create a custom config file:

```json
{
    "epochs": 200,
    "batch_size": 128,
    "lr": 5e-4,
    "cfg_scale": 10.0,
    "target_embedding_dim": 32
}
```

Use it:
```bash
python src/train_all.py --config my_config.json
```

### **GPU Training**
```bash
# Force CUDA usage
python src/train_all.py --device cuda

# Auto-detect (recommended)
python src/train_all.py --device auto
```

### **Reproducible Research**
```bash
# Fixed seed for reproducibility
python src/train_all.py --seed 12345

# Different seeds for multiple runs
python src/train_all.py --seed 42
python src/train_all.py --seed 123
python src/train_all.py --seed 999
```

## 🔍 Monitoring Training

### **Console Output**
```
Training pipeline initialized:
  Models: ['all']
  Device: cuda
  Results: results
  Run ID: 20241201_143022

============================================================
TRAINING ZERO-CONDITIONED MODEL
============================================================
Creating zero-conditioned DDPM...
Loading financial data...
Loaded 2520 days of return data
Date range: 2014-01-02 to 2023-12-29
Creating sequences of length 60...
Created 2461 sequences
Training zero_conditioned...
Epoch   0: Train Loss: 0.123456, Val Loss: 0.123456
Epoch  10: Train Loss: 0.098765, Val Loss: 0.101234
...
```

### **Progress Tracking**
- **Every 10 epochs**: Training and validation loss printed
- **Early stopping**: Automatic detection and stopping
- **Checkpoint saving**: Best model automatically saved

## 🚨 Troubleshooting

### **Common Issues**

#### **Import Errors**
```bash
# Check if all dependencies are installed
pip install -r requirements_llm_refactored.txt

# Verify imports work
python -c "import torch; import sentence_transformers; print('OK')"
```

#### **CUDA Issues**
```bash
# Force CPU training
python src/train_all.py --device cpu

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### **Memory Issues**
```bash
# Reduce batch size
python src/train_all.py --batch-size 32

# Reduce hidden dimension (if supported)
python src/train_all.py --hidden-dim 64
```

#### **Data Path Issues**
```bash
# Specify data path explicitly
python src/train_all.py --data-path /path/to/sp500_data.csv

# Check fallback paths
ls -la data/sp500_data.csv
ls -la ../data/sp500_data.csv
```

### **Performance Optimization**

#### **Training Speed**
- **Use GPU**: `--device cuda` for faster training
- **Increase batch size**: If memory allows
- **Reduce validation frequency**: Modify code if needed

#### **Memory Efficiency**
- **Reduce batch size**: Smaller batches use less memory
- **Gradient accumulation**: Implement if needed for large models
- **Mixed precision**: Enable if supported

## 🔗 Integration with Evaluation

### **Immediate Evaluation**
After training, each model generates a `README_RUN.md` with evaluation instructions:

```bash
# Example from README_RUN.md
python src/evaluate_model.py --model-name explicit_conditioned --run-id 20241201_143022
```

### **Comprehensive Comparison**
Use the comparison framework for side-by-side evaluation:

```bash
python src/comprehensive_comparison_framework.py \
    --results-dir results \
    --run-id 20241201_143022
```

## 📚 Example Workflows

### **1. Quick Test Run**
```bash
# Train all models with minimal epochs
python src/train_all.py --epochs 10 --batch-size 32
```

### **2. Production Training**
```bash
# Full training with custom config
python src/train_all.py \
    --config configs/production_config.json \
    --seed 42 \
    --device auto
```

### **3. Research Experiment**
```bash
# Multiple runs with different seeds
for seed in 42 123 456 789 999; do
    python src/train_all.py --seed $seed --epochs 100
done
```

### **4. Model Comparison**
```bash
# Train specific models for comparison
python src/train_all.py --models explicit llm --epochs 200
```

## 🎉 Success Indicators

### **Training Completion**
- ✅ All selected models trained successfully
- ✅ Checkpoints saved in results directory
- ✅ Training history and metadata recorded
- ✅ README files generated for evaluation

### **Quality Checks**
- ✅ Validation loss decreasing over time
- ✅ Early stopping working (if applicable)
- ✅ Checkpoints loading correctly
- ✅ All output files generated

## 🔮 Future Enhancements

### **Planned Features**
1. **Distributed training**: Multi-GPU support
2. **Advanced scheduling**: Learning rate scheduling options
3. **Experiment tracking**: Integration with MLflow/W&B
4. **Hyperparameter tuning**: Automated hyperparameter optimization
5. **Model compression**: Quantization and pruning support

### **Research Extensions**
1. **Multi-modal conditioning**: Combine multiple conditioning sources
2. **Attention mechanisms**: Replace FiLM with cross-attention
3. **Hierarchical models**: Multi-scale temporal modeling
4. **Uncertainty quantification**: Confidence intervals for samples

## 📞 Support

### **Documentation**
- **This README**: Complete usage guide
- **Code comments**: Inline documentation
- **Example configs**: Template configurations

### **Debugging**
- **Console output**: Detailed training progress
- **Error messages**: Clear error descriptions
- **Log files**: Complete training history

### **Community**
- **GitHub issues**: Report bugs and request features
- **Code review**: Submit improvements and fixes
- **Documentation**: Help improve this guide

---

**🎯 Ready to train your DDPM models?** Start with the quick start examples and build up to your full research pipeline! 🚀
