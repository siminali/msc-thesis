"""
Script to add evaluation code to existing notebooks

This script adds the comprehensive evaluation framework to your existing GARCH, DDPM, and TimeGrad notebooks.
Run this script to automatically add evaluation cells to your notebooks.

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import json
import os

def add_evaluation_to_garch_notebook():
    """Add evaluation code to the GARCH notebook."""
    
    evaluation_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Comprehensive Evaluation Metrics\n",
                "\n",
                "The following cells add comprehensive evaluation metrics as requested by the supervisor.\n",
                "This includes automated metrics, plots, and LaTeX table generation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import evaluation framework\n",
                "import sys\n",
                "sys.path.append('../src')\n",
                "from evaluation_framework import FinancialModelEvaluator\n",
                "\n",
                "# Initialize evaluator\n",
                "evaluator = FinancialModelEvaluator(model_names=['GARCH'])\n",
                "print(\"✅ Evaluation framework loaded!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Prepare GARCH results for evaluation\n",
                "# Assuming you have garch_forecasts and test_returns from earlier cells\n",
                "\n",
                "# Save GARCH results for comprehensive evaluation\n",
                "import numpy as np\n",
                "np.save('../results/garch_returns.npy', test_returns.values)\n",
                "np.save('../results/garch_var_forecasts.npy', garch_forecasts.values)\n",
                "\n",
                "print(\"✅ GARCH results saved for comprehensive evaluation!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run comprehensive evaluation for GARCH\n",
                "real_data = test_returns.values\n",
                "synthetic_data_dict = {'GARCH': test_returns.values}  # Using test returns as synthetic for comparison\n",
                "var_forecasts_dict = {'GARCH': garch_forecasts.values}\n",
                "\n",
                "results = evaluator.run_comprehensive_evaluation(\n",
                "    real_data=real_data,\n",
                "    synthetic_data_dict=synthetic_data_dict,\n",
                "    var_forecasts_dict=var_forecasts_dict,\n",
                "    save_path=\"../results/garch_evaluation/\"\n",
                ")\n",
                "\n",
                "print(\"\\n🎉 GARCH comprehensive evaluation completed!\")\n",
                "print(\"📊 Results saved to: ../results/garch_evaluation/\")"
            ]
        }
    ]
    
    return evaluation_cells

def add_evaluation_to_diffusion_notebook():
    """Add evaluation code to the DDPM notebook."""
    
    evaluation_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Comprehensive Evaluation Metrics\n",
                "\n",
                "The following cells add comprehensive evaluation metrics as requested by the supervisor.\n",
                "This includes automated metrics, plots, and LaTeX table generation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import evaluation framework\n",
                "import sys\n",
                "sys.path.append('../src')\n",
                "from evaluation_framework import FinancialModelEvaluator\n",
                "\n",
                "# Initialize evaluator\n",
                "evaluator = FinancialModelEvaluator(model_names=['DDPM'])\n",
                "print(\"✅ Evaluation framework loaded!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Prepare DDPM results for evaluation\n",
                "# Assuming you have 'samples' from the DDPM generation\n",
                "\n",
                "# Save DDPM results for comprehensive evaluation\n",
                "import numpy as np\n",
                "np.save('../results/ddpm_returns.npy', samples.numpy())\n",
                "\n",
                "print(\"✅ DDPM results saved for comprehensive evaluation!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run comprehensive evaluation for DDPM\n",
                "real_data = X.flatten()  # Flatten the real training data\n",
                "synthetic_data_dict = {'DDPM': samples.numpy().flatten()}\n",
                "\n",
                "results = evaluator.run_comprehensive_evaluation(\n",
                "    real_data=real_data,\n",
                "    synthetic_data_dict=synthetic_data_dict,\n",
                "    save_path=\"../results/ddpm_evaluation/\"\n",
                ")\n",
                "\n",
                "print(\"\\n🎉 DDPM comprehensive evaluation completed!\")\n",
                "print(\"📊 Results saved to: ../results/ddpm_evaluation/\")"
            ]
        }
    ]
    
    return evaluation_cells

def add_evaluation_to_timegrad_notebook():
    """Add evaluation code to the TimeGrad notebook."""
    
    evaluation_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Comprehensive Evaluation Metrics\n",
                "\n",
                "The following cells add comprehensive evaluation metrics as requested by the supervisor.\n",
                "This includes automated metrics, plots, and LaTeX table generation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import evaluation framework\n",
                "import sys\n",
                "sys.path.append('../src')\n",
                "from evaluation_framework import FinancialModelEvaluator\n",
                "\n",
                "# Initialize evaluator\n",
                "evaluator = FinancialModelEvaluator(model_names=['TimeGrad'])\n",
                "print(\"✅ Evaluation framework loaded!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Prepare TimeGrad results for evaluation\n",
                "# Assuming you have 'generated' from the TimeGrad generation\n",
                "\n",
                "# Save TimeGrad results for comprehensive evaluation\n",
                "import numpy as np\n",
                "np.save('../results/timegrad_returns.npy', generated.numpy())\n",
                "\n",
                "print(\"✅ TimeGrad results saved for comprehensive evaluation!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run comprehensive evaluation for TimeGrad\n",
                "real_data = log_returns  # Use the original log returns\n",
                "synthetic_data_dict = {'TimeGrad': generated.numpy().flatten()}\n",
                "\n",
                "results = evaluator.run_comprehensive_evaluation(\n",
                "    real_data=real_data,\n",
                "    synthetic_data_dict=synthetic_data_dict,\n",
                "    save_path=\"../results/timegrad_evaluation/\"\n",
                ")\n",
                "\n",
                "print(\"\\n🎉 TimeGrad comprehensive evaluation completed!\")\n",
                "print(\"📊 Results saved to: ../results/timegrad_evaluation/\")"
            ]
        }
    ]
    
    return evaluation_cells

def add_evaluation_cells_to_notebook(notebook_path, evaluation_cells):
    """Add evaluation cells to a notebook."""
    
    # Read the existing notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Add evaluation cells at the end
    notebook['cells'].extend(evaluation_cells)
    
    # Write back to the notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Added evaluation cells to {notebook_path}")

def main():
    """Main function to add evaluation to all notebooks."""
    
    print("🚀 Adding comprehensive evaluation to all notebooks...")
    
    # Create results directory
    os.makedirs("../results", exist_ok=True)
    
    # Add evaluation to GARCH notebook
    garch_cells = add_evaluation_to_garch_notebook()
    add_evaluation_cells_to_notebook("notebooks/garch.ipynb", garch_cells)
    
    # Add evaluation to DDPM notebook
    ddpm_cells = add_evaluation_to_diffusion_notebook()
    add_evaluation_cells_to_notebook("notebooks/diffusion.ipynb", ddpm_cells)
    
    # Add evaluation to TimeGrad notebook
    timegrad_cells = add_evaluation_to_timegrad_notebook()
    add_evaluation_cells_to_notebook("notebooks/timegrad.ipynb", timegrad_cells)
    
    print("\n🎉 Successfully added evaluation cells to all notebooks!")
    print("\n📝 Next steps:")
    print("1. Run each notebook to execute the evaluation")
    print("2. Check the results in the ../results/ directory")
    print("3. Use the generated LaTeX tables and plots in your thesis")
    print("4. Run the comprehensive_evaluation.ipynb for cross-model comparison")

if __name__ == "__main__":
    main()
