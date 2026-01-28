#!/usr/bin/env python3
"""
Generate publication figures from experiment results.

This script generates the figures used in the paper:
1. Detection rate comparison (bar chart)
2. Detection delay distribution (box plots)
3. CDS time series example
4. Component contribution analysis
5. ROC curves for different thresholds

Usage:
    python experiments/generate_figures.py --input results/experiment_results_*.json
    python experiments/generate_figures.py --input results/ --output figures/
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Publication-quality settings
FIGSIZE_SINGLE = (4, 3)
FIGSIZE_DOUBLE = (8, 3)
FIGSIZE_LARGE = (8, 6)
DPI = 300
FONT_SIZE = 10

# Color scheme (colorblind-friendly)
COLORS = {
    'smf_aas': '#2274A5',   # Blue
    'cusum': '#F75C03',     # Orange
    'adwin': '#D90368',     # Pink
    'perf_only': '#00CC66', # Green
}

METHOD_NAMES = {
    'smf_aas': 'SMF-AAS',
    'cusum': 'CUSUM',
    'adwin': 'ADWIN',
    'perf_only': 'Perf-Only',
}


def setup_matplotlib():
    """Configure matplotlib for publication quality."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'font.family': 'serif',
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE + 1,
        'xtick.labelsize': FONT_SIZE - 1,
        'ytick.labelsize': FONT_SIZE - 1,
        'legend.fontsize': FONT_SIZE - 1,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def load_results(input_path: str) -> Dict[str, Any]:
    """Load experiment results from JSON file(s)."""
    path = Path(input_path)
    
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    
    elif path.is_dir():
        # Find most recent results file
        files = sorted(path.glob("experiment_results_*.json"))
        if not files:
            raise FileNotFoundError(f"No experiment results found in {path}")
        
        with open(files[-1]) as f:
            print(f"Loading: {files[-1]}")
            return json.load(f)
    
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def fig_detection_rates(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate detection rate comparison bar chart (Figure 1)."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    envs = list(results.keys())
    methods = ['smf_aas', 'cusum', 'adwin', 'perf_only']
    
    x = np.arange(len(envs))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    
    for i, method in enumerate(methods):
        rates = []
        for env in envs:
            rate = results[env]['summary'][method]['detection_rate']
            rates.append(rate * 100)
        
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, rates, width, 
                      label=METHOD_NAMES[method],
                      color=COLORS[method],
                      edgecolor='black',
                      linewidth=0.5)
    
    ax.set_ylabel('Detection Rate (%)')
    ax.set_xlabel('Environment')
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in envs])
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(loc='upper right', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_detection_rates.pdf')
    plt.savefig(output_dir / 'fig1_detection_rates.png')
    plt.close()
    print(f"Saved: fig1_detection_rates.pdf")


def fig_detection_delay(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate detection delay box plots (Figure 2)."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    envs = list(results.keys())
    methods = ['smf_aas', 'cusum', 'adwin', 'perf_only']
    
    fig, axes = plt.subplots(1, len(envs), figsize=(10, 3), sharey=True)
    
    for ax_idx, env in enumerate(envs):
        ax = axes[ax_idx]
        
        data = []
        labels = []
        colors = []
        
        for method in methods:
            # Get individual delays
            individual = results[env].get('individual_results', [])
            if method == 'smf_aas':
                delays = [r['detection_delay'] for r in individual 
                         if r['detection_delay'] is not None]
            else:
                delays = [r.get(f'{method}_delay') for r in individual
                         if r.get(f'{method}_delay') is not None]
            
            if delays:
                data.append(delays)
                labels.append(METHOD_NAMES[method])
                colors.append(COLORS[method])
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_title(env.capitalize())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    axes[0].set_ylabel('Detection Delay (episodes)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_detection_delay.pdf')
    plt.savefig(output_dir / 'fig2_detection_delay.png')
    plt.close()
    print(f"Saved: fig2_detection_delay.pdf")


def fig_false_positives(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate false positive comparison (Figure 3)."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    envs = list(results.keys())
    methods = ['smf_aas', 'cusum', 'adwin', 'perf_only']
    
    x = np.arange(len(envs))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    
    for i, method in enumerate(methods):
        fps = []
        errs = []
        for env in envs:
            fp = results[env]['summary'][method]['mean_fp']
            fp_std = results[env]['summary'][method].get('std_fp', 0)
            fps.append(fp)
            errs.append(fp_std)
        
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, fps, width, yerr=errs,
                      label=METHOD_NAMES[method],
                      color=COLORS[method],
                      edgecolor='black',
                      linewidth=0.5,
                      capsize=2)
    
    ax.set_ylabel('False Positives')
    ax.set_xlabel('Environment')
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in envs])
    ax.legend(loc='upper right', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_false_positives.pdf')
    plt.savefig(output_dir / 'fig3_false_positives.png')
    plt.close()
    print(f"Saved: fig3_false_positives.pdf")


def fig_summary_table(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate summary comparison table (Table 1)."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    methods = ['smf_aas', 'cusum', 'adwin', 'perf_only']
    
    # Aggregate across environments
    agg = {m: {'det': [], 'delay': [], 'fp': []} for m in methods}
    
    for env, data in results.items():
        for method in methods:
            s = data['summary'][method]
            agg[method]['det'].append(s['detection_rate'])
            if s['mean_delay'] is not None:
                agg[method]['delay'].append(s['mean_delay'])
            agg[method]['fp'].append(s['mean_fp'])
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    
    cell_text = []
    for method in methods:
        det = np.mean(agg[method]['det']) * 100
        delay = np.mean(agg[method]['delay']) if agg[method]['delay'] else float('nan')
        fp = np.mean(agg[method]['fp'])
        
        cell_text.append([
            METHOD_NAMES[method],
            f"{det:.0f}%",
            f"{delay:.1f}" if not np.isnan(delay) else "N/A",
            f"{fp:.1f}"
        ])
    
    table = ax.table(
        cellText=cell_text,
        colLabels=['Method', 'Det. Rate', 'Mean Delay', 'Mean FP'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZE)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'table1_summary.pdf')
    plt.savefig(output_dir / 'table1_summary.png')
    plt.close()
    print(f"Saved: table1_summary.pdf")


def generate_all_figures(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate all paper figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating figures in: {output_dir}")
    print("-" * 50)
    
    fig_detection_rates(results, output_dir)
    fig_detection_delay(results, output_dir)
    fig_false_positives(results, output_dir)
    fig_summary_table(results, output_dir)
    
    print("-" * 50)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication figures from experiment results"
    )
    parser.add_argument(
        "--input", type=str, default="results",
        help="Input results file or directory"
    )
    parser.add_argument(
        "--output", type=str, default="results/figures",
        help="Output directory for figures"
    )
    
    args = parser.parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for figure generation")
        print("Install with: pip install matplotlib")
        sys.exit(1)
    
    setup_matplotlib()
    results = load_results(args.input)
    generate_all_figures(results, Path(args.output))


if __name__ == "__main__":
    main()
