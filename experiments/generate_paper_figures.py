"""
Generate publication-quality figures for SMF-AAS paper.

Creates:
1. Detection rate comparison (bar chart)
2. Detection delay comparison (bar chart)  
3. False positive comparison (bar chart)
4. Time series of CDS during experiment
5. Component contribution analysis

Usage:
    python generate_paper_figures.py --results results/experiment_results.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color scheme (colorblind-friendly)
COLORS = {
    'smf_aas': '#2ecc71',    # Green
    'cusum': '#3498db',       # Blue
    'adwin': '#9b59b6',       # Purple
    'perf_only': '#e74c3c',   # Red
}

METHOD_NAMES = {
    'smf_aas': 'SMF-AAS (Ours)',
    'cusum': 'CUSUM',
    'adwin': 'ADWIN',
    'perf_only': 'Perf-Only',
}

ENV_NAMES = {
    'tictactoe': 'TicTacToe',
    'connectfour': 'ConnectFour',
    'kuhnpoker': 'KuhnPoker',
    'maze': 'Maze',
}


def load_results(filepath: str) -> Dict:
    """Load experiment results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def figure1_detection_rates(results: Dict, output_dir: Path) -> None:
    """Create detection rate comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    envs = list(results.keys())
    methods = ['smf_aas', 'cusum', 'adwin', 'perf_only']
    n_envs = len(envs)
    n_methods = len(methods)
    
    bar_width = 0.2
    x = np.arange(n_envs)
    
    for i, method in enumerate(methods):
        rates = []
        for env in envs:
            rate = results[env]['summary'][method]['detection_rate']
            rates.append(rate * 100)  # Convert to percentage
        
        offset = (i - n_methods/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, rates, bar_width, 
                     label=METHOD_NAMES[method], color=COLORS[method],
                     edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{rate:.0f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Environment')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Detection Rate Comparison Across Methods')
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_NAMES.get(e, e) for e in envs])
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right')
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_detection_rates.pdf')
    plt.savefig(output_dir / 'figure1_detection_rates.png')
    plt.close()
    print(f"  Saved: figure1_detection_rates.pdf/png")


def figure2_detection_delays(results: Dict, output_dir: Path) -> None:
    """Create detection delay comparison bar chart with error bars."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    envs = list(results.keys())
    methods = ['smf_aas', 'cusum', 'adwin', 'perf_only']
    n_envs = len(envs)
    n_methods = len(methods)
    
    bar_width = 0.2
    x = np.arange(n_envs)
    
    for i, method in enumerate(methods):
        delays = []
        errors = []
        for env in envs:
            summary = results[env]['summary'][method]
            delay = summary['mean_delay']
            std = summary['std_delay']
            if delay is None:
                delays.append(0)
                errors.append(0)
            else:
                delays.append(delay)
                errors.append(std)
        
        offset = (i - n_methods/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, delays, bar_width,
                     label=METHOD_NAMES[method], color=COLORS[method],
                     edgecolor='black', linewidth=0.5,
                     yerr=errors, capsize=3, error_kw={'linewidth': 1})
    
    ax.set_xlabel('Environment')
    ax.set_ylabel('Detection Delay (episodes)')
    ax.set_title('Detection Delay Comparison (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_NAMES.get(e, e) for e in envs])
    ax.legend(loc='upper right')
    
    # Add note about N/A values
    ax.text(0.02, 0.98, 'Height=0 indicates no detection', 
            transform=ax.transAxes, fontsize=8, va='top', style='italic')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_detection_delays.pdf')
    plt.savefig(output_dir / 'figure2_detection_delays.png')
    plt.close()
    print(f"  Saved: figure2_detection_delays.pdf/png")


def figure3_false_positives(results: Dict, output_dir: Path) -> None:
    """Create false positive comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    envs = list(results.keys())
    methods = ['smf_aas', 'cusum', 'adwin', 'perf_only']
    n_envs = len(envs)
    n_methods = len(methods)
    
    bar_width = 0.2
    x = np.arange(n_envs)
    
    for i, method in enumerate(methods):
        fps = []
        errors = []
        for env in envs:
            summary = results[env]['summary'][method]
            fp = summary['mean_fp']
            std = summary['std_fp']
            fps.append(fp)
            errors.append(std)
        
        offset = (i - n_methods/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, fps, bar_width,
                     label=METHOD_NAMES[method], color=COLORS[method],
                     edgecolor='black', linewidth=0.5,
                     yerr=errors, capsize=3, error_kw={'linewidth': 1})
    
    ax.set_xlabel('Environment')
    ax.set_ylabel('False Positives (count)')
    ax.set_title('False Positive Comparison (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_NAMES.get(e, e) for e in envs])
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_false_positives.pdf')
    plt.savefig(output_dir / 'figure3_false_positives.png')
    plt.close()
    print(f"  Saved: figure3_false_positives.pdf/png")


def figure4_summary_table(results: Dict, output_dir: Path) -> None:
    """Create summary comparison as a figure (for paper appendix)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Build table data
    methods = ['smf_aas', 'cusum', 'adwin', 'perf_only']
    envs = list(results.keys())
    
    # Headers
    col_labels = ['Environment'] + [METHOD_NAMES[m] for m in methods]
    
    # Data rows
    table_data = []
    for env in envs:
        row = [ENV_NAMES.get(env, env)]
        for method in methods:
            s = results[env]['summary'][method]
            rate = s['detection_rate'] * 100
            delay = s['mean_delay']
            fp = s['mean_fp']
            
            if delay is not None:
                cell = f"{rate:.0f}% ({delay:.0f}±{s['std_delay']:.0f})\nFP: {fp:.1f}"
            else:
                cell = f"{rate:.0f}%\n(no detection)"
            row.append(cell)
        table_data.append(row)
    
    # Add aggregate row
    agg_row = ['Overall']
    for method in methods:
        rates = [results[env]['summary'][method]['detection_rate'] for env in envs]
        delays = [results[env]['summary'][method]['mean_delay'] for env in envs 
                  if results[env]['summary'][method]['mean_delay'] is not None]
        fps = [results[env]['summary'][method]['mean_fp'] for env in envs]
        
        avg_rate = np.mean(rates) * 100
        avg_delay = np.mean(delays) if delays else None
        avg_fp = np.mean(fps)
        
        if avg_delay is not None:
            cell = f"{avg_rate:.0f}% ({avg_delay:.1f})\nFP: {avg_fp:.1f}"
        else:
            cell = f"{avg_rate:.0f}%"
        agg_row.append(cell)
    table_data.append(agg_row)
    
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)
    
    # Color the header
    for i, label in enumerate(col_labels):
        table[(0, i)].set_facecolor('#4a4a4a')
        table[(0, i)].set_text_props(color='white', weight='bold')
    
    # Highlight SMF-AAS column
    for i in range(len(table_data) + 1):
        if i > 0:
            table[(i, 1)].set_facecolor('#d5f5e3')
    
    # Highlight overall row
    for j in range(len(col_labels)):
        table[(len(table_data), j)].set_facecolor('#f5f5f5')
        table[(len(table_data), j)].set_text_props(weight='bold')
    
    plt.title('Summary: Detection Rate (Delay±Std) and False Positives', 
              fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_summary_table.pdf')
    plt.savefig(output_dir / 'figure4_summary_table.png')
    plt.close()
    print(f"  Saved: figure4_summary_table.pdf/png")


def figure5_radar_chart(results: Dict, output_dir: Path) -> None:
    """Create radar chart comparing methods across metrics."""
    from math import pi
    
    methods = ['smf_aas', 'cusum', 'adwin', 'perf_only']
    envs = list(results.keys())
    
    # Metrics: detection rate, 1/delay (normalized), 1/FP (normalized)
    metrics = ['Detection\nRate', 'Speed\n(1/delay)', 'Precision\n(1/FP)']
    n_metrics = len(metrics)
    
    # Compute aggregate scores for each method
    scores = {}
    for method in methods:
        rates = [results[env]['summary'][method]['detection_rate'] for env in envs]
        delays = [results[env]['summary'][method]['mean_delay'] for env in envs 
                  if results[env]['summary'][method]['mean_delay'] is not None]
        fps = [results[env]['summary'][method]['mean_fp'] for env in envs]
        
        det_rate = np.mean(rates)
        
        # Invert delay (higher = better), normalize to [0,1]
        if delays:
            avg_delay = np.mean(delays)
            speed = 1.0 / (1.0 + avg_delay / 50)  # Normalize
        else:
            speed = 0
        
        # Invert FP (lower = better), normalize to [0,1]
        avg_fp = np.mean(fps)
        precision = 1.0 / (1.0 + avg_fp)
        
        scores[method] = [det_rate, speed, precision]
    
    # Create radar chart
    angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for method in methods:
        values = scores[method]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=METHOD_NAMES[method],
                color=COLORS[method])
        ax.fill(angles, values, alpha=0.25, color=COLORS[method])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.title('Method Comparison Across Metrics', size=14, y=1.08)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure5_radar_chart.pdf')
    plt.savefig(output_dir / 'figure5_radar_chart.png')
    plt.close()
    print(f"  Saved: figure5_radar_chart.pdf/png")


def figure6_maze_highlight(results: Dict, output_dir: Path) -> None:
    """Create figure highlighting SMF-AAS advantage on Maze."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Detection rates on Maze only
    ax1 = axes[0]
    methods = ['smf_aas', 'cusum', 'adwin', 'perf_only']
    rates = [results['maze']['summary'][m]['detection_rate'] * 100 for m in methods]
    
    bars = ax1.bar(range(len(methods)), rates, 
                   color=[COLORS[m] for m in methods],
                   edgecolor='black', linewidth=1)
    
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([METHOD_NAMES[m] for m in methods], rotation=15)
    ax1.set_ylabel('Detection Rate (%)')
    ax1.set_title('Maze Environment: Detection Rate')
    ax1.set_ylim(0, 100)
    
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    # Highlight SMF-AAS
    bars[0].set_edgecolor('gold')
    bars[0].set_linewidth(3)
    
    # Right: Explanation text
    ax2 = axes[1]
    ax2.axis('off')
    
    explanation = """
    Why SMF-AAS succeeds on Maze:
    
    • Maze is a single-player navigation task
    • Strategy change = goal position moved
    • No performance change (agent still fails initially)
    
    Baseline methods fail because:
    • CUSUM/ADWIN: Monitor scalar performance only
    • Performance-Only: No reward change detected
    
    SMF-AAS succeeds because:
    • S component: Detects changed state visitation
    • B component: Detects changed movement patterns
    • Multi-component approach captures behavioral shift
      even without performance change
    
    This demonstrates SMF-AAS's advantage for detecting
    strategy changes that don't immediately affect
    observable performance metrics.
    """
    
    ax2.text(0.1, 0.9, explanation, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    ax2.set_title('Analysis: Why Multi-Component Detection Matters')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure6_maze_analysis.pdf')
    plt.savefig(output_dir / 'figure6_maze_analysis.png')
    plt.close()
    print(f"  Saved: figure6_maze_analysis.pdf/png")


def generate_latex_table(results: Dict, output_dir: Path) -> None:
    """Generate LaTeX table for paper."""
    methods = ['smf_aas', 'cusum', 'adwin', 'perf_only']
    envs = list(results.keys())
    
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Detection performance comparison across environments. " +
                 r"Rate shows detection percentage, Delay shows mean episodes $\pm$ std, " +
                 r"FP shows mean false positives. Best results in \textbf{bold}.}")
    latex.append(r"\label{tab:results}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{l|ccc|ccc|ccc|ccc}")
    latex.append(r"\toprule")
    latex.append(r"& \multicolumn{3}{c|}{SMF-AAS (Ours)} & \multicolumn{3}{c|}{CUSUM} & " +
                 r"\multicolumn{3}{c|}{ADWIN} & \multicolumn{3}{c}{Perf-Only} \\")
    latex.append(r"Env & Rate & Delay & FP & Rate & Delay & FP & Rate & Delay & FP & Rate & Delay & FP \\")
    latex.append(r"\midrule")
    
    for env in envs:
        row = [ENV_NAMES.get(env, env)]
        
        # Find best values for highlighting
        all_rates = [results[env]['summary'][m]['detection_rate'] for m in methods]
        all_delays = [results[env]['summary'][m]['mean_delay'] for m in methods]
        all_fps = [results[env]['summary'][m]['mean_fp'] for m in methods]
        
        best_rate = max(all_rates)
        valid_delays = [d for d in all_delays if d is not None]
        best_delay = min(valid_delays) if valid_delays else None
        best_fp = min(all_fps)
        
        for method in methods:
            s = results[env]['summary'][method]
            
            # Rate
            rate = s['detection_rate'] * 100
            rate_str = f"{rate:.0f}\\%"
            if rate == best_rate * 100:
                rate_str = r"\textbf{" + rate_str + "}"
            row.append(rate_str)
            
            # Delay
            delay = s['mean_delay']
            if delay is not None:
                delay_str = f"{delay:.0f}"
                if delay == best_delay:
                    delay_str = r"\textbf{" + delay_str + "}"
            else:
                delay_str = "--"
            row.append(delay_str)
            
            # FP
            fp = s['mean_fp']
            fp_str = f"{fp:.1f}"
            if fp == best_fp:
                fp_str = r"\textbf{" + fp_str + "}"
            row.append(fp_str)
        
        latex.append(" & ".join(row) + r" \\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    # Write to file
    with open(output_dir / 'table_results.tex', 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"  Saved: table_results.tex")


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to experiment results JSON')
    parser.add_argument('--output', type=str, default='results/figures',
                       help='Output directory for figures')
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results}")
    results = load_results(args.results)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Generate figures
    print("\nGenerating figures...")
    figure1_detection_rates(results, output_dir)
    figure2_detection_delays(results, output_dir)
    figure3_false_positives(results, output_dir)
    figure4_summary_table(results, output_dir)
    figure5_radar_chart(results, output_dir)
    figure6_maze_highlight(results, output_dir)
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    generate_latex_table(results, output_dir)
    
    print("\n✓ All figures generated successfully!")
    print(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    main()
