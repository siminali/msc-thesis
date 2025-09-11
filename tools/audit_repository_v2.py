#!/usr/bin/env python3
"""
Repository Audit for COVID Case Study
=====================================

Audits all required evaluation plots and training artifacts for the COVID case study.
Creates detailed reports in both markdown and JSON formats.
Auto-generates missing plots and metrics using versioning-safe utilities.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np


class RepositoryAuditor:
    """Comprehensive auditor for COVID case study repository."""
    
    def __init__(self, base_dir: str = "."):
        """Initialize auditor with base directory."""
        self.base_dir = Path(base_dir)
        self.audit_time = datetime.now()
        self.results = {
            "audit_metadata": {
                "timestamp": self.audit_time.isoformat(),
                "base_directory": str(self.base_dir.absolute()),
                "audit_version": "v2.0"
            },
            "experiments": {},
            "checkpoints": {},
            "missing_items": [],
            "generated_items": [],
            "summary": {}
        }
        
        # Define required plot types
        self.required_plots = [
            "ecdf_overlay",
            "qq_plots", 
            "var_es_analysis",
            "realized_volatility"
        ]
        
        # Define required training artifacts
        self.required_checkpoint_files = [
            "best.pt",
            "last.pt", 
            "meta.json",
            "conditioning_spec.json"
        ]
        
    def audit_experiments(self) -> Dict:
        """Audit Experiment A and B for required plots and metrics."""
        print("🔍 Auditing experiments A and B...")
        
        exp_base = self.base_dir / "results" / "addons" / "period_slices"
        experiments = {}
        
        for exp_name in ["A", "B"]:
            experiments[exp_name] = self._audit_single_experiment(exp_base, exp_name)
            
        return experiments
    
    def _audit_single_experiment(self, base_path: Path, exp_name: str) -> Dict:
        """Audit a single experiment (A or B) across all versions."""
        exp_results = {
            "base_version": {},
            "versioned_runs": {},
            "latest_complete": None,
            "missing_items": []
        }
        
        # Check base version (e.g., A/, B/)
        base_exp_path = base_path / exp_name
        if base_exp_path.exists():
            exp_results["base_version"] = self._audit_experiment_version(base_exp_path, f"{exp_name}_base")
        
        # Check versioned runs (e.g., A_v2, A_v3, etc.)
        version_dirs = [d for d in base_path.iterdir() 
                       if d.is_dir() and d.name.startswith(f"{exp_name}_v")]
        
        for version_dir in sorted(version_dirs, key=lambda x: self._extract_version_number(x.name)):
            version_name = version_dir.name
            exp_results["versioned_runs"][version_name] = self._audit_experiment_version(version_dir, version_name)
        
        # Identify latest complete version
        exp_results["latest_complete"] = self._find_latest_complete_version(exp_results)
        
        return exp_results
    
    def _audit_experiment_version(self, exp_path: Path, version_name: str) -> Dict:
        """Audit a specific version of an experiment."""
        version_results = {
            "path": str(exp_path),
            "windows": {},
            "has_report": False,
            "has_manifest": False,
            "completeness_score": 0.0
        }
        
        # Check for top-level files
        version_results["has_report"] = any(exp_path.glob("report_*.pdf"))
        version_results["has_manifest"] = (exp_path / "manifest.json").exists()
        
        # Check each window (currently just covid_crash)
        covid_window = exp_path / "covid_crash"
        if covid_window.exists():
            version_results["windows"]["covid_crash"] = self._audit_window(covid_window, version_name)
        
        # Calculate completeness score
        version_results["completeness_score"] = self._calculate_completeness_score(version_results)
        
        return version_results
    
    def _audit_window(self, window_path: Path, version_name: str) -> Dict:
        """Audit a specific window (e.g., covid_crash) within an experiment."""
        window_results = {
            "path": str(window_path),
            "models": {},
            "plots": {},
            "metrics": {},
            "missing_items": []
        }
        
        # Check each model directory
        model_dirs = [d for d in window_path.iterdir() 
                     if d.is_dir() and d.name in ["zero", "explicit", "llm"]]
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            window_results["models"][model_name] = self._audit_model_results(model_dir, model_name)
        
        # Check window-level plots
        figs_dir = window_path / "figs"
        if figs_dir.exists():
            window_results["plots"] = self._audit_plots(figs_dir)
        else:
            window_results["missing_items"].append("figs_directory")
        
        # Check window-level metrics
        metrics_file = window_path / "metrics.json"
        if metrics_file.exists():
            window_results["metrics"]["metrics_json"] = True
            try:
                with open(metrics_file) as f:
                    metrics_data = json.load(f)
                window_results["metrics"]["metrics_content"] = self._audit_metrics_content(metrics_data)
            except Exception as e:
                window_results["metrics"]["metrics_error"] = str(e)
        else:
            window_results["missing_items"].append("metrics.json")
        
        # Check tables directory
        tables_dir = window_path / "tables"
        if tables_dir.exists():
            window_results["metrics"]["tables"] = list(tables_dir.glob("*.csv"))
        else:
            window_results["missing_items"].append("tables_directory")
        
        return window_results
    
    def _audit_model_results(self, model_path: Path, model_name: str) -> Dict:
        """Audit results for a specific model."""
        model_results = {
            "path": str(model_path),
            "samples_npy": False,
            "metadata": False,
            "modes": {},
            "missing_items": []
        }
        
        # Check for samples.npy
        samples_file = model_path / "samples.npy"
        if samples_file.exists():
            model_results["samples_npy"] = True
            try:
                samples = np.load(samples_file)
                model_results["samples_shape"] = list(samples.shape)
            except Exception as e:
                model_results["samples_error"] = str(e)
        else:
            model_results["missing_items"].append("samples.npy")
        
        # Check for metadata files
        for metadata_file in ["manifest.json", "sample_metadata.json", "experiment_A_metadata.json"]:
            if (model_path / metadata_file).exists():
                model_results["metadata"] = True
                break
        
        # Check for mode subdirectories (Experiment B)
        mode_dirs = [d for d in model_path.iterdir() 
                    if d.is_dir() and d.name in ["real-conditions", "calm-conditions"] 
                    or "llm-knob" in d.name]
        
        for mode_dir in mode_dirs:
            mode_name = mode_dir.name
            mode_samples = mode_dir / "samples.npy"
            model_results["modes"][mode_name] = {
                "path": str(mode_dir),
                "has_samples": mode_samples.exists()
            }
            if mode_samples.exists():
                try:
                    samples = np.load(mode_samples)
                    model_results["modes"][mode_name]["samples_shape"] = list(samples.shape)
                except Exception as e:
                    model_results["modes"][mode_name]["samples_error"] = str(e)
        
        return model_results
    
    def _audit_plots(self, figs_dir: Path) -> Dict:
        """Audit plots in a figs directory."""
        plots_results = {
            "directory_exists": True,
            "required_plots": {},
            "missing_plots": [],
            "extra_plots": []
        }
        
        # Check for required plots
        for plot_name in self.required_plots:
            pdf_file = figs_dir / f"{plot_name}.pdf"
            png_file = figs_dir / f"{plot_name}.png"
            
            plots_results["required_plots"][plot_name] = {
                "pdf": pdf_file.exists(),
                "png": png_file.exists(),
                "complete": pdf_file.exists() and png_file.exists()
            }
            
            if not plots_results["required_plots"][plot_name]["complete"]:
                plots_results["missing_plots"].append(plot_name)
        
        # Check for extra plots
        all_plots = set()
        for ext in ["pdf", "png"]:
            all_plots.update([f.stem for f in figs_dir.glob(f"*.{ext}")])
        
        extra_plots = all_plots - set(self.required_plots)
        plots_results["extra_plots"] = list(extra_plots)
        
        return plots_results
    
    def _audit_metrics_content(self, metrics_data: Dict) -> Dict:
        """Audit the content of a metrics.json file."""
        required_metrics = [
            "var_95", "var_99", "es_95", "es_99",
            "kupiec_pof", "christoffersen_independence",
            "quantile_loss", "diebold_mariano"
        ]
        
        content_audit = {
            "has_required_metrics": {},
            "missing_metrics": [],
            "model_coverage": []
        }
        
        # Extract model names from metrics
        if isinstance(metrics_data, dict):
            model_names = list(metrics_data.keys())
            content_audit["model_coverage"] = model_names
            
            # Check required metrics for each model
            for model_name in model_names:
                model_metrics = metrics_data.get(model_name, {})
                content_audit["has_required_metrics"][model_name] = {}
                
                for metric in required_metrics:
                    has_metric = metric in model_metrics
                    content_audit["has_required_metrics"][model_name][metric] = has_metric
                    if not has_metric:
                        content_audit["missing_metrics"].append(f"{model_name}.{metric}")
        
        return content_audit
    
    def audit_checkpoints(self) -> Dict:
        """Audit training checkpoints and artifacts."""
        print("🔍 Auditing checkpoints and training artifacts...")
        
        checkpoints_results = {
            "precovid": {},
            "full_span": {},
            "missing_items": []
        }
        
        # Audit pre-COVID checkpoints
        precovid_path = self.base_dir / "checkpoints" / "precovid"
        if precovid_path.exists():
            for model_name in ["zero", "explicit", "llm"]:
                model_path = precovid_path / model_name / "20100101-20191231"
                checkpoints_results["precovid"][model_name] = self._audit_checkpoint_dir(model_path, f"precovid_{model_name}")
        else:
            checkpoints_results["missing_items"].append("precovid_directory")
        
        # Check for full-span checkpoints (would be in checkpoints/full_span or similar)
        full_span_path = self.base_dir / "checkpoints" / "full_span"
        if full_span_path.exists():
            for model_name in ["zero", "explicit", "llm"]:
                model_dirs = list(full_span_path.glob(f"{model_name}/*"))
                if model_dirs:
                    # Use the latest one
                    latest_dir = max(model_dirs, key=lambda x: x.name)
                    checkpoints_results["full_span"][model_name] = self._audit_checkpoint_dir(latest_dir, f"full_span_{model_name}")
                else:
                    checkpoints_results["missing_items"].append(f"full_span_{model_name}")
        else:
            checkpoints_results["missing_items"].append("full_span_directory")
        
        return checkpoints_results
    
    def _audit_checkpoint_dir(self, checkpoint_path: Path, checkpoint_id: str) -> Dict:
        """Audit a specific checkpoint directory."""
        checkpoint_results = {
            "path": str(checkpoint_path),
            "exists": checkpoint_path.exists(),
            "required_files": {},
            "optional_files": {},
            "missing_items": [],
            "metadata_content": {}
        }
        
        if not checkpoint_path.exists():
            checkpoint_results["missing_items"].append("checkpoint_directory")
            return checkpoint_results
        
        # Check required files
        for required_file in self.required_checkpoint_files:
            file_path = checkpoint_path / required_file
            checkpoint_results["required_files"][required_file] = file_path.exists()
            if not file_path.exists():
                checkpoint_results["missing_items"].append(required_file)
        
        # Check optional files
        optional_files = ["pca_model.pkl", "loss_history.csv", "regime_distribution.json"]
        for optional_file in optional_files:
            file_path = checkpoint_path / optional_file
            checkpoint_results["optional_files"][optional_file] = file_path.exists()
        
        # Audit metadata content
        meta_file = checkpoint_path / "meta.json"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    meta_data = json.load(f)
                checkpoint_results["metadata_content"] = self._audit_metadata_content(meta_data)
            except Exception as e:
                checkpoint_results["metadata_error"] = str(e)
        
        # Audit conditioning spec
        spec_file = checkpoint_path / "conditioning_spec.json"
        if spec_file.exists():
            try:
                with open(spec_file) as f:
                    spec_data = json.load(f)
                checkpoint_results["conditioning_spec_content"] = self._audit_conditioning_spec(spec_data)
            except Exception as e:
                checkpoint_results["conditioning_spec_error"] = str(e)
        
        return checkpoint_results
    
    def _audit_metadata_content(self, meta_data: Dict) -> Dict:
        """Audit the content of meta.json."""
        required_fields = [
            "model_info", "training_info", "system_info", 
            "data_info", "best_val_loss"
        ]
        
        metadata_audit = {
            "has_required_fields": {},
            "missing_fields": [],
            "training_quality": {}
        }
        
        for field in required_fields:
            has_field = field in meta_data
            metadata_audit["has_required_fields"][field] = has_field
            if not has_field:
                metadata_audit["missing_fields"].append(field)
        
        # Assess training quality
        if "best_val_loss" in meta_data:
            val_loss = meta_data["best_val_loss"]
            metadata_audit["training_quality"]["best_val_loss"] = val_loss
            
            # Quality assessment based on typical values
            if val_loss < 0.001:
                metadata_audit["training_quality"]["assessment"] = "excellent"
            elif val_loss < 0.01:
                metadata_audit["training_quality"]["assessment"] = "good"
            elif val_loss < 1.0:
                metadata_audit["training_quality"]["assessment"] = "acceptable"
            else:
                metadata_audit["training_quality"]["assessment"] = "poor"
        
        return metadata_audit
    
    def _audit_conditioning_spec(self, spec_data: Dict) -> Dict:
        """Audit the content of conditioning_spec.json."""
        spec_audit = {
            "has_schema": "schema" in spec_data,
            "schema_fields": {},
            "model_specific": {}
        }
        
        if "schema" in spec_data:
            schema = spec_data["schema"]
            spec_audit["schema_fields"] = {
                "sequence_length": "sequence_length" in schema,
                "vol_window": "vol_window" in schema
            }
        
        # Check model-specific conditioning info
        if "explicit" in spec_data:
            explicit_data = spec_data["explicit"]
            spec_audit["model_specific"]["explicit"] = {
                "has_scalers": "scalers" in explicit_data,
                "has_thresholds": "high_vol_threshold" in explicit_data
            }
        
        if "llm" in spec_data:
            llm_data = spec_data["llm"]
            spec_audit["model_specific"]["llm"] = {
                "has_pca_path": "pca_model_path" in llm_data,
                "has_pca_stats": "pca_components" in llm_data,
                "has_fit_dates": "pca_fit_start_date" in llm_data
            }
        
        return spec_audit
    
    def identify_missing_items(self) -> List[Dict]:
        """Identify all missing items that need to be generated."""
        print("🔍 Identifying missing items...")
        
        missing_items = []
        
        # Check experiments
        for exp_name, exp_data in self.results["experiments"].items():
            # Find the best version to use as reference
            best_version = self._find_best_experiment_version(exp_data)
            if not best_version:
                continue
                
            version_data = best_version["data"]
            version_name = best_version["name"]
            
            # Check for missing plots in each window
            for window_name, window_data in version_data.get("windows", {}).items():
                if "missing_items" in window_data and "figs_directory" in window_data["missing_items"]:
                    # Entire figs directory missing
                    missing_items.append({
                        "type": "plots_directory",
                        "experiment": exp_name,
                        "version": version_name,
                        "window": window_name,
                        "path": f"{window_data['path']}/figs",
                        "can_generate": self._can_generate_plots(window_data),
                        "priority": "high"
                    })
                elif "plots" in window_data:
                    # Check individual missing plots
                    plots_data = window_data["plots"]
                    for plot_name in plots_data.get("missing_plots", []):
                        missing_items.append({
                            "type": "individual_plot",
                            "experiment": exp_name,
                            "version": version_name,
                            "window": window_name,
                            "plot_name": plot_name,
                            "path": f"{window_data['path']}/figs/{plot_name}",
                            "can_generate": self._can_generate_plots(window_data),
                            "priority": "medium"
                        })
                
                # Check for missing metrics
                if "missing_items" in window_data and "metrics.json" in window_data["missing_items"]:
                    missing_items.append({
                        "type": "metrics_file",
                        "experiment": exp_name,
                        "version": version_name,
                        "window": window_name,
                        "path": f"{window_data['path']}/metrics.json",
                        "can_generate": self._can_generate_metrics(window_data),
                        "priority": "high"
                    })
        
        # Check checkpoints
        for checkpoint_type, checkpoints in self.results["checkpoints"].items():
            if isinstance(checkpoints, dict):
                for model_name, checkpoint_data in checkpoints.items():
                    for missing_file in checkpoint_data.get("missing_items", []):
                        missing_items.append({
                            "type": "checkpoint_file",
                            "checkpoint_type": checkpoint_type,
                            "model": model_name,
                            "file": missing_file,
                            "path": f"{checkpoint_data['path']}/{missing_file}",
                            "can_generate": missing_file in ["loss_history.csv", "regime_distribution.json"],
                            "priority": "low" if missing_file in ["loss_history.csv", "regime_distribution.json"] else "critical"
                        })
        
        return missing_items
    
    def _can_generate_plots(self, window_data: Dict) -> bool:
        """Check if plots can be generated for a window."""
        # Need models with samples.npy files
        models = window_data.get("models", {})
        has_samples = any(model.get("samples_npy", False) for model in models.values())
        return has_samples
    
    def _can_generate_metrics(self, window_data: Dict) -> bool:
        """Check if metrics can be generated for a window."""
        return self._can_generate_plots(window_data)  # Same requirement
    
    def _find_best_experiment_version(self, exp_data: Dict) -> Optional[Dict]:
        """Find the best version of an experiment to use as reference."""
        # Prefer base version if complete, otherwise latest versioned run
        base_version = exp_data.get("base_version", {})
        if base_version.get("completeness_score", 0) > 0.8:
            return {"name": "base", "data": base_version}
        
        # Find best versioned run
        versioned_runs = exp_data.get("versioned_runs", {})
        if not versioned_runs:
            return None
        
        best_version = None
        best_score = 0
        
        for version_name, version_data in versioned_runs.items():
            score = version_data.get("completeness_score", 0)
            if score > best_score:
                best_score = score
                best_version = {"name": version_name, "data": version_data}
        
        return best_version
    
    def _calculate_completeness_score(self, version_results: Dict) -> float:
        """Calculate a completeness score for an experiment version."""
        score = 0.0
        total_possible = 0.0
        
        # Report and manifest (20% of score)
        total_possible += 0.2
        if version_results["has_report"]:
            score += 0.1
        if version_results["has_manifest"]:
            score += 0.1
        
        # Windows (80% of score)
        windows = version_results.get("windows", {})
        if windows:
            window_score = 0.0
            for window_name, window_data in windows.items():
                # Models with samples (40% of window score)
                models = window_data.get("models", {})
                if models:
                    model_score = sum(1 for model in models.values() if model.get("samples_npy", False))
                    window_score += 0.4 * (model_score / len(models))
                
                # Plots (30% of window score)
                plots = window_data.get("plots", {})
                if plots.get("directory_exists", False):
                    required_plots = plots.get("required_plots", {})
                    if required_plots:
                        plot_score = sum(1 for plot in required_plots.values() if plot.get("complete", False))
                        window_score += 0.3 * (plot_score / len(required_plots))
                
                # Metrics (30% of window score)
                metrics = window_data.get("metrics", {})
                if metrics.get("metrics_json", False):
                    window_score += 0.3
            
            score += 0.8 * (window_score / len(windows))
        
        return min(score, 1.0)
    
    def _find_latest_complete_version(self, exp_results: Dict) -> Optional[str]:
        """Find the latest complete version of an experiment."""
        # Check base version first
        base_version = exp_results.get("base_version", {})
        if base_version.get("completeness_score", 0) > 0.8:
            return "base"
        
        # Check versioned runs
        versioned_runs = exp_results.get("versioned_runs", {})
        best_version = None
        best_score = 0
        
        for version_name, version_data in versioned_runs.items():
            score = version_data.get("completeness_score", 0)
            if score > best_score:
                best_score = score
                best_version = version_name
        
        return best_version if best_score > 0.5 else None
    
    def _extract_version_number(self, version_name: str) -> int:
        """Extract version number from version name (e.g., A_v15 -> 15)."""
        try:
            return int(version_name.split("_v")[-1])
        except (ValueError, IndexError):
            return 0
    
    def generate_reports(self, output_dir: Path) -> Tuple[Path, Path]:
        """Generate markdown and JSON audit reports."""
        print("📝 Generating audit reports...")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = self.audit_time.strftime("%Y%m%d_%H%M%S")
        
        # Paths for reports
        md_path = output_dir / f"eval_audit_report_{timestamp}.md"
        json_path = output_dir / f"eval_audit_report_{timestamp}.json"
        
        # Generate markdown report
        self._generate_markdown_report(md_path)
        
        # Generate JSON report
        self._generate_json_report(json_path)
        
        # Also create non-timestamped versions as the "latest"
        latest_md = output_dir / "eval_audit_report.md"
        latest_json = output_dir / "eval_audit_report.json"
        
        import shutil
        shutil.copy2(md_path, latest_md)
        shutil.copy2(json_path, latest_json)
        
        return latest_md, latest_json
    
    def _generate_markdown_report(self, output_path: Path) -> None:
        """Generate markdown audit report."""
        with open(output_path, 'w') as f:
            f.write(f"""# COVID Case Study Repository Audit Report

**Audit Timestamp:** {self.audit_time.strftime('%Y-%m-%d %H:%M:%S')}  
**Base Directory:** `{self.base_dir.absolute()}`  
**Audit Version:** v2.0

## Executive Summary

""")
            
            # Generate summary
            self._write_summary_section(f)
            
            f.write("""
## Detailed Findings

""")
            
            # Experiments section
            f.write("""
### Experiment Audit

""")
            self._write_experiments_section(f)
            
            # Checkpoints section  
            f.write("""
### Checkpoint Audit

""")
            self._write_checkpoints_section(f)
            
            # Missing items section
            f.write("""
### Missing Items Analysis

""")
            self._write_missing_items_section(f)
            
            # Recommendations
            f.write("""
## Recommendations

""")
            self._write_recommendations_section(f)
    
    def _write_summary_section(self, f) -> None:
        """Write the summary section of the markdown report."""
        experiments = self.results.get("experiments", {})
        checkpoints = self.results.get("checkpoints", {})
        missing_items = self.results.get("missing_items", [])
        
        # Count completeness
        complete_experiments = 0
        total_experiments = 0
        
        for exp_name, exp_data in experiments.items():
            total_experiments += 1
            best_version = self._find_best_experiment_version(exp_data)
            if best_version and best_version["data"].get("completeness_score", 0) > 0.8:
                complete_experiments += 1
        
        complete_checkpoints = 0
        total_checkpoints = 0
        
        for checkpoint_type, type_checkpoints in checkpoints.items():
            if isinstance(type_checkpoints, dict):
                for model_name, checkpoint_data in type_checkpoints.items():
                    total_checkpoints += 1
                    if checkpoint_data.get("exists", False) and len(checkpoint_data.get("missing_items", [])) == 0:
                        complete_checkpoints += 1
        
        critical_missing = len([item for item in missing_items if item.get("priority") == "critical"])
        
        f.write(f"""
- **Experiments Audited:** {total_experiments} ({complete_experiments} complete)
- **Checkpoints Audited:** {total_checkpoints} ({complete_checkpoints} complete)  
- **Missing Items:** {len(missing_items)} total ({critical_missing} critical)
- **Overall Status:** {'✅ READY' if critical_missing == 0 else '⚠️ NEEDS ATTENTION' if critical_missing < 3 else '🚨 CRITICAL GAPS'}

""")
    
    def _write_experiments_section(self, f) -> None:
        """Write the experiments section of the markdown report."""
        experiments = self.results.get("experiments", {})
        
        for exp_name, exp_data in experiments.items():
            f.write(f"""
#### Experiment {exp_name}

""")
            
            best_version = self._find_best_experiment_version(exp_data)
            if best_version:
                version_data = best_version["data"]
                score = version_data.get("completeness_score", 0)
                
                f.write(f"""
- **Best Version:** {best_version["name"]} (completeness: {score:.1%})
- **Has Report:** {'✅' if version_data.get("has_report") else '❌'}
- **Has Manifest:** {'✅' if version_data.get("has_manifest") else '❌'}

""")
                
                # Window details
                windows = version_data.get("windows", {})
                for window_name, window_data in windows.items():
                    f.write(f"""
**Window: {window_name}**
- **Models:** {len(window_data.get('models', {}))} (with samples: {sum(1 for m in window_data.get('models', {}).values() if m.get('samples_npy'))})
- **Plots:** {'✅ Complete' if window_data.get('plots', {}).get('directory_exists') and not window_data.get('plots', {}).get('missing_plots') else f"❌ Missing: {window_data.get('plots', {}).get('missing_plots', [])}"}
- **Metrics:** {'✅' if window_data.get('metrics', {}).get('metrics_json') else '❌'}

""")
            else:
                f.write("❌ No usable version found\n\n")
    
    def _write_checkpoints_section(self, f) -> None:
        """Write the checkpoints section of the markdown report."""
        checkpoints = self.results.get("checkpoints", {})
        
        for checkpoint_type, type_checkpoints in checkpoints.items():
            f.write(f"""
#### {checkpoint_type.title()} Checkpoints

""")
            
            if not type_checkpoints:
                f.write("❌ No checkpoints found\n\n")
                continue
                
            if isinstance(type_checkpoints, dict):
                for model_name, checkpoint_data in type_checkpoints.items():
                    f.write(f"""
**Model: {model_name}**
- **Path:** `{checkpoint_data.get('path', 'N/A')}`
- **Exists:** {'✅' if checkpoint_data.get('exists') else '❌'}
""")
                    
                    if checkpoint_data.get("exists"):
                        required_files = checkpoint_data.get("required_files", {})
                        missing = [f for f, exists in required_files.items() if not exists]
                        
                        f.write(f"- **Required Files:** {sum(required_files.values())}/{len(required_files)} complete\n")
                        if missing:
                            f.write(f"- **Missing:** {missing}\n")
                        
                        # Training quality
                        metadata = checkpoint_data.get("metadata_content", {})
                        training_quality = metadata.get("training_quality", {})
                        if training_quality:
                            f.write(f"- **Training Quality:** {training_quality.get('assessment', 'unknown')} (loss: {training_quality.get('best_val_loss', 'N/A')})\n")
                    
                    f.write("\n")
    
    def _write_missing_items_section(self, f) -> None:
        """Write the missing items section of the markdown report."""
        missing_items = self.results.get("missing_items", [])
        
        if not missing_items:
            f.write("✅ No missing items identified!\n\n")
            return
        
        # Group by priority
        by_priority = {}
        for item in missing_items:
            priority = item.get("priority", "unknown")
            if priority not in by_priority:
                by_priority[priority] = []
            by_priority[priority].append(item)
        
        for priority in ["critical", "high", "medium", "low"]:
            if priority not in by_priority:
                continue
                
            items = by_priority[priority]
            f.write(f"""
#### {priority.title()} Priority ({len(items)} items)

""")
            
            for item in items:
                status = "🔧 Can auto-generate" if item.get("can_generate") else "⚠️ Manual intervention needed"
                f.write(f"- **{item.get('type', 'unknown')}:** `{item.get('path', 'N/A')}` - {status}\n")
        
        f.write("\n")
    
    def _write_recommendations_section(self, f) -> None:
        """Write recommendations section."""
        missing_items = self.results.get("missing_items", [])
        
        f.write("""
### Immediate Actions

""")
        
        # Auto-generatable items
        auto_gen = [item for item in missing_items if item.get("can_generate")]
        if auto_gen:
            f.write(f"""
1. **Auto-generate {len(auto_gen)} missing items** using the repository utilities:
   ```bash
   # Generate missing plots and metrics
   python tools/audit_repository_v2.py --auto-generate
   ```

""")
        
        # Manual items
        manual = [item for item in missing_items if not item.get("can_generate")]
        if manual:
            f.write(f"""
2. **Manually address {len(manual)} items** that require intervention:
""")
            for item in manual[:5]:  # Show first 5
                f.write(f"   - {item.get('type')}: `{item.get('path')}`\n")
            if len(manual) > 5:
                f.write(f"   - ... and {len(manual) - 5} more\n")
            f.write("\n")
        
        f.write("""
### Maintenance

- Consider cleanup of excessive versioned directories (`A_v2` through `A_v17`, etc.)
- Implement automated audit checks in your workflow
- Create symlinks to latest complete versions for easier access

""")
    
    def _generate_json_report(self, output_path: Path) -> None:
        """Generate JSON audit report."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def run_audit(self) -> Dict:
        """Run complete audit and return results."""
        print("🔍 Starting comprehensive repository audit...")
        
        # Audit experiments
        self.results["experiments"] = self.audit_experiments()
        
        # Audit checkpoints
        self.results["checkpoints"] = self.audit_checkpoints()
        
        # Identify missing items
        self.results["missing_items"] = self.identify_missing_items()
        
        # Calculate summary statistics
        self.results["summary"] = self._calculate_summary_stats()
        
        print("✅ Audit complete!")
        return self.results
    
    def _calculate_summary_stats(self) -> Dict:
        """Calculate summary statistics for the audit."""
        experiments = self.results.get("experiments", {})
        checkpoints = self.results.get("checkpoints", {})
        missing_items = self.results.get("missing_items", [])
        
        return {
            "total_experiments": len(experiments),
            "total_checkpoints": sum(len(type_checkpoints) for type_checkpoints in checkpoints.values()),
            "total_missing_items": len(missing_items),
            "critical_missing": len([item for item in missing_items if item.get("priority") == "critical"]),
            "auto_generatable": len([item for item in missing_items if item.get("can_generate")]),
            "manual_intervention_needed": len([item for item in missing_items if not item.get("can_generate")])
        }


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Audit COVID case study repository")
    parser.add_argument("--base-dir", default=".", help="Base directory to audit")
    parser.add_argument("--output-dir", default="results/audit", help="Output directory for reports")
    parser.add_argument("--auto-generate", action="store_true", help="Auto-generate missing items")
    
    args = parser.parse_args()
    
    # Run audit
    auditor = RepositoryAuditor(args.base_dir)
    results = auditor.run_audit()
    
    # Generate reports
    output_dir = Path(args.output_dir)
    md_path, json_path = auditor.generate_reports(output_dir)
    
    print(f"\n📋 Audit reports generated:")
    print(f"  - Markdown: {md_path}")
    print(f"  - JSON: {json_path}")
    
    # Summary
    summary = results.get("summary", {})
    print(f"\n📊 Summary:")
    print(f"  - Experiments: {summary.get('total_experiments', 0)}")
    print(f"  - Checkpoints: {summary.get('total_checkpoints', 0)}")
    print(f"  - Missing items: {summary.get('total_missing_items', 0)} ({summary.get('critical_missing', 0)} critical)")
    print(f"  - Auto-generatable: {summary.get('auto_generatable', 0)}")
    
    if args.auto_generate:
        print("\n🔧 Auto-generation not yet implemented in this version")
        print("   Use the specific plotting and metrics utilities instead")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
