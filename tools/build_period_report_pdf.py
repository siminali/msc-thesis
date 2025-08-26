#!/usr/bin/env python3
"""
Build a standalone PDF that compiles the period-sliced report.

Primary path (A): use LaTeX (pdflatex/xelatex) if available.
Fallback (B): use reportlab to stitch figures and render tables.

Output: results/addons/period_slices/period_report.pdf
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from typing import List
import csv


def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def build_with_latex(base_dir: str, section_tex: str, out_pdf: str) -> bool:
    tex_cmd = 'pdflatex' if which('pdflatex') else ('xelatex' if which('xelatex') else None)
    if tex_cmd is None:
        return False
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        main_tex = os.path.join(tmp, 'main.tex')
        # Minimal preamble
        preamble = ['\\documentclass[11pt]{article}',
                    '\\usepackage[margin=1in]{geometry}',
                    '\\usepackage{graphicx}',
                    '\\usepackage{booktabs}',
                    '\\usepackage{caption}',
                    '\\title{Period-Sliced Evaluation Report}',
                    f'\\date{{}}',
                    '\\begin{document}',
                    '\\maketitle',
                    f'\\input{{{os.path.abspath(section_tex)}}}',
                    '\\end{document}']
        with open(main_tex, 'w') as f:
            f.write('\n'.join(preamble))
        # Run twice for references
        for _ in range(2):
            subprocess.run([tex_cmd, '-interaction=nonstopmode', os.path.basename(main_tex)], cwd=tmp, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        built_pdf = os.path.join(tmp, 'main.pdf')
        if os.path.exists(built_pdf):
            shutil.copyfile(built_pdf, out_pdf)
            return True
    return False


def build_with_reportlab(base_dir: str, out_pdf: str) -> bool:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import Table, TableStyle
    except Exception:
        return False

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    c = canvas.Canvas(out_pdf, pagesize=A4)
    width, height = A4

    # Title page
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-100, "Period-Sliced Evaluation Report")
    c.setFont("Helvetica", 12)
    c.drawString(72, height-140, f"Base directory: {base_dir}")
    c.showPage()

    # Iterate windows and drop in key figures (PDF/PNG)
    for w in sorted(os.listdir(base_dir)):
        wdir = os.path.join(base_dir, w)
        if not os.path.isdir(wdir):
            continue
        figs = os.path.join(wdir, 'figures')
        if not os.path.isdir(figs):
            continue
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height-72, f"Window: {w}")
        y = height - 110
        # Place up to 3 figures per page
        placed = 0
        for fig in sorted(os.listdir(figs)):
            if not fig.lower().endswith(('.png', '.pdf', '.jpg', '.jpeg')):
                continue
            path = os.path.join(figs, fig)
            try:
                img = ImageReader(path)
                iw, ih = img.getSize()
                max_w = width - 2*72
                max_h = 3.5*inch
                scale = min(max_w/iw, max_h/ih)
                draw_w, draw_h = iw*scale, ih*scale
                if y - draw_h < 72:
                    c.showPage()
                    c.setFont("Helvetica-Bold", 16)
                    c.drawString(72, height-72, f"Window: {w} (cont.)")
                    y = height - 110
                c.drawImage(img, 72, y-draw_h, width=draw_w, height=draw_h)
                y -= (draw_h + 24)
                placed += 1
            except Exception:
                continue
        # Try to render a compact per-window metrics table
        metrics_csv_path = os.path.join(wdir, 'metrics.csv')
        if os.path.exists(metrics_csv_path):
            try:
                with open(metrics_csv_path, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                if rows:
                    header = rows[0]
                    data_rows = rows[1:13]
                    table_data = [header] + data_rows
                    tbl = Table(table_data, hAlign='LEFT')
                    tbl.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
                        ('FONTSIZE', (0, 0), (-1, -1), 8),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ]))
                    tw, th = tbl.wrapOn(c, width - 144, height)
                    if y - th < 72:
                        c.showPage()
                        c.setFont("Helvetica-Bold", 16)
                        c.drawString(72, height-72, f"Window: {w} (tables)")
                        y = height - 110
                    tbl.drawOn(c, 72, y - th)
                    y -= (th + 24)
            except Exception:
                pass
        c.showPage()
    # Add cross-window summary table if available
    summary_csv_path = os.path.join(base_dir, 'summary.csv')
    if os.path.exists(summary_csv_path):
        try:
            c.setFont("Helvetica-Bold", 16)
            c.drawString(72, height-72, "Cross-window Summary")
            y = height - 110
            with open(summary_csv_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            if rows:
                header = rows[0]
                data_rows = rows[1:]
                table_data = [header] + data_rows
                tbl = Table(table_data, hAlign='LEFT')
                tbl.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ]))
                tw, th = tbl.wrapOn(c, width - 144, height)
                if y - th < 72:
                    c.showPage()
                    c.setFont("Helvetica-Bold", 16)
                    c.drawString(72, height-72, "Cross-window Summary (cont.)")
                    y = height - 110
                tbl.drawOn(c, 72, y - th)
                c.showPage()
        except Exception:
            pass
    c.save()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description='Build period report PDF from per-window artifacts.')
    parser.add_argument('--base', default='results/addons/period_slices', help='Base period slices directory')
    parser.add_argument('--section', default='results/addons/period_slices/period_report_section.tex', help='Section .tex path')
    parser.add_argument('--out', default='results/addons/period_slices/period_report.pdf', help='Output PDF path')
    args = parser.parse_args()

    # Attempt LaTeX build first
    if build_with_latex(args.base, args.section, args.out):
        return
    # Fallback to reportlab stitching
    if build_with_reportlab(args.base, args.out):
        return
    raise SystemExit('Failed to build report: LaTeX not available and reportlab fallback failed.')


if __name__ == '__main__':
    main()


