#!/usr/bin/env python3
"""
Report Compiler: Professional PDF Report Generation

Generates comprehensive PDF reports from experiment results including:
- Overview tables with model performance metrics
- Key figures (ECDF, Q-Q plots, VaR/ES analysis, volatility tracking)
- Findings paragraphs extracted from metrics and findings files
- Professional layout with consistent formatting
- Robust handling of missing assets (red "SKIPPED" boxes)

Features:
- Automatic figure inclusion from figs/ directories
- Metrics extraction from metrics.json files
- Findings synthesis from findings.jsonl files
- Multi-window and multi-mode support (Experiments A & B)
- Never fails build - graceful degradation for missing assets
- Professional PDF layout with proper typography

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

# PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import red, black, blue, darkblue, gray
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReportCompiler:
    """Compiles comprehensive PDF reports from experiment results."""
    
    def __init__(self, experiment_dir: Path):
        """Initialize the report compiler."""
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = self.experiment_dir.name
        
        # Output paths
        self.output_pdf = self.experiment_dir / f"report_{self.experiment_name}.pdf"
        
        # Initialize PDF components
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Data containers
        self.windows = {}
        self.findings = []
        self.overall_summary = {}
        
        logger.info(f"Initialized ReportCompiler for experiment: {self.experiment_name}")
        logger.info(f"Output PDF: {self.output_pdf}")
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=20,
            textColor=darkblue,
            alignment=TA_CENTER,
            spaceAfter=30
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=darkblue,
            spaceBefore=20,
            spaceAfter=12
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=blue,
            spaceBefore=15,
            spaceAfter=8
        ))
        
        # Findings style
        self.styles.add(ParagraphStyle(
            name='Findings',
            parent=self.styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceBefore=8,
            spaceAfter=8,
            leftIndent=20,
            rightIndent=20
        ))
        
        # Skipped style
        self.styles.add(ParagraphStyle(
            name='Skipped',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=red,
            alignment=TA_CENTER,
            backColor=colors.pink,
            spaceBefore=10,
            spaceAfter=10
        ))
    
    def discover_windows(self):
        """Discover all windows in the experiment directory."""
        try:
            if not self.experiment_dir.exists():
                logger.error(f"Experiment directory does not exist: {self.experiment_dir}")
                return
            
            # Look for window directories
            window_dirs = [d for d in self.experiment_dir.iterdir() 
                          if d.is_dir() and not d.name.startswith('.')]
            
            # Filter out common non-window directories
            exclude_dirs = {'figures', 'figs', 'tables', 'cache', 'logs'}
            window_dirs = [d for d in window_dirs if d.name not in exclude_dirs]
            
            for window_dir in window_dirs:
                window_id = window_dir.name
                # Try to find figs directory - might be in base experiment directory
                figs_dir = window_dir / 'figs'
                if not figs_dir.exists():
                    # Check base experiment directory
                    base_exp_name = self.experiment_name.split('_')[0]
                    base_exp_dir = self.experiment_dir.parent / base_exp_name
                    alt_figs_dir = base_exp_dir / window_id / 'figs'
                    if alt_figs_dir.exists():
                        figs_dir = alt_figs_dir
                
                self.windows[window_id] = {
                    'directory': window_dir,
                    'metrics_file': window_dir / 'metrics.json',
                    'figs_dir': figs_dir,
                    'models': {},
                    'modes': {}  # For Experiment B
                }
                
                # Discover models/modes in this window
                self._discover_window_contents(window_id)
            
            logger.info(f"Discovered {len(self.windows)} windows: {list(self.windows.keys())}")
            
        except Exception as e:
            logger.error(f"Error discovering windows: {e}")
    
    def _discover_window_contents(self, window_id: str):
        """Discover models and modes within a window."""
        window_dir = self.windows[window_id]['directory']
        
        # Look for model directories
        for item in window_dir.iterdir():
            if item.is_dir() and item.name not in {'figs', 'tables'}:
                # Check if this is a model directory (has samples.npy)
                if (item / 'samples.npy').exists():
                    self.windows[window_id]['models'][item.name] = {
                        'directory': item,
                        'samples_file': item / 'samples.npy',
                        'metadata_file': item / 'sample_metadata.json'
                    }
                else:
                    # This might be a mode directory (Experiment B)
                    # Look for subdirectories with samples
                    for subitem in item.iterdir():
                        if subitem.is_dir() and (subitem / 'samples.npy').exists():
                            mode_key = f"{item.name}_{subitem.name}"
                            self.windows[window_id]['modes'][mode_key] = {
                                'model': item.name,
                                'mode': subitem.name,
                                'directory': subitem,
                                'samples_file': subitem / 'samples.npy',
                                'metadata_file': subitem / 'sample_metadata.json'
                            }
    
    def load_findings(self):
        """Load findings from findings.jsonl files."""
        try:
            findings_files = list(self.experiment_dir.glob('**/findings.jsonl'))
            
            for findings_file in findings_files:
                logger.info(f"Loading findings from: {findings_file}")
                try:
                    with open(findings_file, 'r') as f:
                        for line in f:
                            finding = json.loads(line.strip())
                            self.findings.append(finding)
                except Exception as e:
                    logger.warning(f"Error loading findings from {findings_file}: {e}")
            
            logger.info(f"Loaded {len(self.findings)} findings entries")
            
        except Exception as e:
            logger.error(f"Error loading findings: {e}")
    
    def load_window_metrics(self, window_id: str) -> Optional[Dict[str, Any]]:
        """Load metrics for a specific window."""
        try:
            metrics_file = self.windows[window_id]['metrics_file']
            
            # If not found in versioned directory, try base experiment directory
            if not metrics_file.exists():
                # Extract base experiment name (A from A_v8, B from B_v3)
                base_exp_name = self.experiment_name.split('_')[0]
                base_exp_dir = self.experiment_dir.parent / base_exp_name
                alt_metrics_file = base_exp_dir / window_id / 'metrics.json'
                
                if alt_metrics_file.exists():
                    logger.info(f"Found metrics in base directory: {alt_metrics_file}")
                    metrics_file = alt_metrics_file
                else:
                    logger.warning(f"Metrics file not found in {metrics_file} or {alt_metrics_file}")
                    return None
            
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            logger.info(f"Loaded metrics for window: {window_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error loading metrics for {window_id}: {e}")
            return None
    
    def create_skipped_element(self, asset_name: str, width: float = 6*inch, height: float = 2*inch):
        """Create a red SKIPPED box for missing assets."""
        from reportlab.platypus import Flowable
        from reportlab.lib.colors import red, white
        
        class SkippedBox(Flowable):
            def __init__(self, asset_name, width, height):
                self.asset_name = asset_name
                self.width = width
                self.height = height
            
            def draw(self):
                # Draw red border box
                self.canv.setStrokeColor(red)
                self.canv.setFillColor(white)
                self.canv.setLineWidth(2)
                self.canv.rect(0, 0, self.width, self.height, fill=1)
                
                # Draw diagonal lines
                self.canv.setStrokeColor(red)
                self.canv.line(0, 0, self.width, self.height)
                self.canv.line(0, self.height, self.width, 0)
                
                # Add SKIPPED text
                self.canv.setFillColor(red)
                self.canv.setFont("Helvetica-Bold", 16)
                text_width = self.canv.stringWidth("SKIPPED", "Helvetica-Bold", 16)
                x = (self.width - text_width) / 2
                y = self.height / 2 + 10
                self.canv.drawString(x, y, "SKIPPED")
                
                # Add asset name
                self.canv.setFont("Helvetica", 10)
                asset_text = f"Missing: {self.asset_name}"
                text_width = self.canv.stringWidth(asset_text, "Helvetica", 10)
                x = (self.width - text_width) / 2
                y = self.height / 2 - 10
                self.canv.drawString(x, y, asset_text)
        
        return SkippedBox(asset_name, width, height)
    
    def create_overview_table(self, window_id: str, metrics: Dict[str, Any]) -> Table:
        """Create overview table for a window."""
        try:
            # Prepare table data
            headers = ['Model', 'VaR (5%)', 'ES (5%)', 'Volatility', 'Status']
            table_data = [headers]
            
            # Add model rows
            models = metrics.get('models', {})
            for model_name, model_data in models.items():
                if model_data.get('status') == 'success':
                    risk_metrics = model_data.get('risk_metrics', {})
                    basic_stats = model_data.get('basic_stats', {})
                    
                    var_5 = risk_metrics.get('var_0.050', 'N/A')
                    es_5 = risk_metrics.get('es_0.050', 'N/A')
                    volatility = basic_stats.get('std', 'N/A')
                    
                    # Format numbers
                    if isinstance(var_5, (int, float)):
                        var_5 = f"{var_5:.3f}"
                    if isinstance(es_5, (int, float)):
                        es_5 = f"{es_5:.3f}"
                    if isinstance(volatility, (int, float)):
                        volatility = f"{volatility:.3f}"
                    
                    row = [model_name, var_5, es_5, volatility, 'Success']
                else:
                    row = [model_name, 'Failed', 'Failed', 'Failed', 'Failed']
                
                table_data.append(row)
            
            # Create table
            table = Table(table_data, colWidths=[1.2*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            return table
            
        except Exception as e:
            logger.error(f"Error creating overview table for {window_id}: {e}")
            return self.create_skipped_element(f"Overview table for {window_id}")
    
    def add_figure_or_skipped(self, elements: List, fig_path: Path, caption: str, width: float = 5*inch):
        """Add figure to elements, or SKIPPED box if missing."""
        try:
            if fig_path.exists():
                # Add figure
                img = Image(str(fig_path), width=width, height=width*0.75)
                elements.append(img)
                
                # Add caption
                caption_para = Paragraph(f"<i>{caption}</i>", self.styles['Normal'])
                elements.append(caption_para)
            else:
                # Add SKIPPED box
                skipped = self.create_skipped_element(f"Figure: {fig_path.name}")
                elements.append(skipped)
                
                # Add missing caption
                caption_para = Paragraph(f"<i>Missing: {caption}</i>", self.styles['Skipped'])
                elements.append(caption_para)
            
            elements.append(Spacer(1, 12))
            
        except Exception as e:
            logger.error(f"Error adding figure {fig_path}: {e}")
            skipped = self.create_skipped_element(f"Error loading: {fig_path.name}")
            elements.append(skipped)
            elements.append(Spacer(1, 12))
    
    def create_findings_paragraph(self, window_id: str) -> Paragraph:
        """Create findings paragraph for a window."""
        try:
            # Find findings for this window
            window_findings = [f for f in self.findings if f.get('window_id') == window_id]
            
            if not window_findings:
                return Paragraph(
                    "No findings available for this window.", 
                    self.styles['Skipped']
                )
            
            # Use the latest finding
            finding = window_findings[-1]
            
            # Build findings text
            findings_text = []
            
            # Model performance
            models = finding.get('models', {})
            if models:
                findings_text.append("Model Performance:")
                for model_name, metrics in models.items():
                    var_5 = metrics.get('var_5pct', 'N/A')
                    if isinstance(var_5, (int, float)):
                        var_5 = f"{var_5:.3f}"
                    findings_text.append(f"• {model_name}: VaR(5%) = {var_5}")
            
            # Pairwise comparisons
            comparisons = finding.get('pairwise_comparisons', {})
            if comparisons:
                findings_text.append("\nModel Comparisons:")
                for comparison_name, stats in comparisons.items():
                    dm_p = stats.get('dm_mse_pvalue', 'N/A')
                    if isinstance(dm_p, (int, float)):
                        significance = "significant" if dm_p < 0.05 else "not significant"
                        findings_text.append(f"• {comparison_name}: {significance} difference (p={dm_p:.4f})")
            
            # Controllability insights (Experiment B)
            controllability = finding.get('controllability_insights', {})
            if controllability:
                findings_text.append("\nControllability:")
                var_diff = controllability.get('var_5pct_diff_pct')
                conservative = controllability.get('more_conservative')
                if var_diff is not None and conservative:
                    findings_text.append(f"• {conservative} model is {abs(var_diff):.1f}% more conservative")
            
            # Join findings
            full_text = "\n".join(findings_text) if findings_text else "No detailed findings available."
            
            return Paragraph(full_text, self.styles['Findings'])
            
        except Exception as e:
            logger.error(f"Error creating findings for {window_id}: {e}")
            return Paragraph(
                f"Error loading findings for {window_id}: {str(e)}", 
                self.styles['Skipped']
            )
    
    def compile_window_section(self, window_id: str) -> List:
        """Compile a complete section for one window."""
        elements = []
        
        # Window header
        header = Paragraph(f"Window: {window_id}", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 12))
        
        # Load metrics
        metrics = self.load_window_metrics(window_id)
        
        if metrics:
            # Overview table
            overview_header = Paragraph("Performance Overview", self.styles['SubsectionHeader'])
            elements.append(overview_header)
            
            overview_table = self.create_overview_table(window_id, metrics)
            elements.append(overview_table)
            elements.append(Spacer(1, 20))
            
            # Key figures
            figures_header = Paragraph("Key Figures", self.styles['SubsectionHeader'])
            elements.append(figures_header)
            
            figs_dir = self.windows[window_id]['figs_dir']
            
            # Use PNG files for images (ReportLab works better with PNG than PDF)
            # ECDF plot
            ecdf_path = figs_dir / 'ecdf_overlay.png'
            self.add_figure_or_skipped(elements, ecdf_path, f"ECDF Overlay - {window_id}")
            
            # Q-Q plots
            qq_path = figs_dir / 'qq_plots.png'
            self.add_figure_or_skipped(elements, qq_path, f"Q-Q Plots - {window_id}")
            
            # VaR/ES analysis
            var_path = figs_dir / 'var_es_analysis.png'
            self.add_figure_or_skipped(elements, var_path, f"VaR/ES Analysis - {window_id}")
            
            # Volatility tracking
            vol_path = figs_dir / 'realized_volatility.png'
            self.add_figure_or_skipped(elements, vol_path, f"Volatility Tracking - {window_id}")
            
            # Findings
            findings_header = Paragraph("Key Findings", self.styles['SubsectionHeader'])
            elements.append(findings_header)
            
            findings_para = self.create_findings_paragraph(window_id)
            elements.append(findings_para)
            
        else:
            # No metrics available
            skipped = self.create_skipped_element(f"Metrics for {window_id}")
            elements.append(skipped)
        
        # Add page break after each window (except last)
        elements.append(PageBreak())
        
        return elements
    
    def create_report_title(self) -> List:
        """Create the report title page."""
        elements = []
        
        # Main title
        title_text = f"Experiment {self.experiment_name.upper()} Report"
        title = Paragraph(title_text, self.styles['CustomTitle'])
        elements.append(title)
        elements.append(Spacer(1, 30))
        
        # Experiment description
        if self.experiment_name.upper() == 'A':
            description = """
            <b>Out-of-Sample Stress Testing</b><br/>
            Testing pre-COVID trained models on stress periods (COVID crash, recovery, etc.)
            using models trained only on pre-COVID data (2010-2019).
            """
        elif self.experiment_name.upper() == 'B':
            description = """
            <b>Counterfactual Controllability Testing</b><br/>
            Testing the controllability of pre-COVID models by manipulating conditioning inputs
            while keeping model weights fixed across different scenarios.
            """
        else:
            description = f"<b>Experiment {self.experiment_name.upper()}</b><br/>Comprehensive analysis report."
        
        desc_para = Paragraph(description, self.styles['Normal'])
        elements.append(desc_para)
        elements.append(Spacer(1, 40))
        
        # Report metadata
        metadata_text = f"""
        <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Experiment Directory:</b> {self.experiment_dir}<br/>
        <b>Windows Analyzed:</b> {len(self.windows)}<br/>
        <b>Total Findings:</b> {len(self.findings)}
        """
        
        metadata_para = Paragraph(metadata_text, self.styles['Normal'])
        elements.append(metadata_para)
        elements.append(Spacer(1, 40))
        
        # Summary table
        if self.windows:
            summary_header = Paragraph("Windows Summary", self.styles['SubsectionHeader'])
            elements.append(summary_header)
            
            summary_data = [['Window', 'Models', 'Modes', 'Status']]
            for window_id, window_info in self.windows.items():
                models_count = len(window_info['models'])
                modes_count = len(window_info['modes'])
                status = 'Ready' if window_info['metrics_file'].exists() else 'Missing Metrics'
                summary_data.append([window_id, str(models_count), str(modes_count), status])
            
            summary_table = Table(summary_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.5*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(summary_table)
        
        elements.append(PageBreak())
        return elements
    
    def compile_report(self):
        """Compile the complete PDF report."""
        try:
            logger.info("Starting report compilation...")
            
            # Discover content
            self.discover_windows()
            self.load_findings()
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(self.output_pdf),
                pagesize=A4,
                topMargin=1*inch,
                bottomMargin=1*inch,
                leftMargin=0.75*inch,
                rightMargin=0.75*inch
            )
            
            # Build story
            story = []
            
            # Title page
            story.extend(self.create_report_title())
            
            # Window sections
            for window_id in sorted(self.windows.keys()):
                logger.info(f"Compiling section for window: {window_id}")
                story.extend(self.compile_window_section(window_id))
            
            # Remove last page break
            if story and isinstance(story[-1], PageBreak):
                story.pop()
            
            # Build PDF
            logger.info(f"Building PDF: {self.output_pdf}")
            doc.build(story)
            
            logger.info(f"Report compiled successfully: {self.output_pdf}")
            
            # Report statistics
            file_size = self.output_pdf.stat().st_size / 1024 / 1024  # MB
            logger.info(f"Report size: {file_size:.2f} MB")
            logger.info(f"Pages generated for {len(self.windows)} windows")
            
            return True
            
        except Exception as e:
            logger.error(f"Error compiling report: {e}")
            return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Report Compiler: Generate PDF reports from experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile Experiment A report
  python report_compiler.py --expdir results/addons/period_slices/A
  
  # Compile Experiment B report
  python report_compiler.py --expdir results/addons/period_slices/B
  
  # Use specific versioned directory
  python report_compiler.py --expdir results/addons/period_slices/A_v8
        """
    )
    
    parser.add_argument('--expdir', type=str, required=True,
                       help='Path to experiment directory (e.g., results/addons/period_slices/A)')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        # Validate experiment directory
        expdir = Path(args.expdir)
        if not expdir.exists():
            logger.error(f"Experiment directory does not exist: {expdir}")
            return
        
        if not expdir.is_dir():
            logger.error(f"Path is not a directory: {expdir}")
            return
        
        # Initialize compiler
        compiler = ReportCompiler(expdir)
        
        # Compile report
        success = compiler.compile_report()
        
        if success:
            print("\n" + "="*60)
            print("REPORT COMPILATION SUMMARY")
            print("="*60)
            print(f"Experiment: {compiler.experiment_name}")
            print(f"Source Directory: {compiler.experiment_dir}")
            print(f"Output PDF: {compiler.output_pdf}")
            print(f"Windows Processed: {len(compiler.windows)}")
            print(f"Findings Included: {len(compiler.findings)}")
            
            if compiler.output_pdf.exists():
                file_size = compiler.output_pdf.stat().st_size / 1024 / 1024
                print(f"Report Size: {file_size:.2f} MB")
            
            print("\nReport compiled successfully! 📄✨")
        else:
            print("❌ Report compilation failed. Check logs for details.")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Report compiler failed: {e}")
        raise

if __name__ == "__main__":
    main()
