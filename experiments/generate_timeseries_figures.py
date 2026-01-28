"""
Generate time-series visualization of monitoring signals.

Creates figures showing how CDS and component scores evolve over time,
with clear indication of the change point and detection.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional

from smf_aas import StrategyMonitor, MonitorConfig, get_env
from smf_aas.agents import TabularQLearning

# Publication settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def run_monitored_experiment(
    env_name: str,
    n_episodes: int = 1500,
    change_point: int = 900,
    baseline_episodes: int = 200,
    window_size: int = 50,
    seed: int = 42
) -> Dict[str, Any]:
    """Run experiment and collect detailed monitoring data."""
    
    env = get_env(env_name)
    opps = env.get_opponents()
    opp_list = list(opps.values())
    is_single = env.num_players == 1
    
    # Train agent
    agent = TabularQLearning(env.num_actions, epsilon=0.15, seed=seed)
    for _ in range(1500):
        if is_single:
            opp_list[0].get_action(np.zeros(env.state_shape), [0])
            if hasattr(opp_list[0], 'reset'):
                opp_list[0].reset()
        
        state = env.reset()
        agent.reset()
        while not state.is_terminal:
            if is_single or state.current_player == 0:
                action = agent.get_action(state.observation, state.legal_actions)
                next_state, _ = env.step(action)
                ret = next_state.returns[0] if next_state.is_terminal and next_state.returns else 0.0
                agent.update(ret, next_state.observation,
                           next_state.legal_actions if not next_state.is_terminal else [], 
                           next_state.is_terminal)
            else:
                action = opp_list[0].get_action(state.observation, state.legal_actions)
                next_state, _ = env.step(action)
            state = next_state
    
    agent.epsilon = 0.05
    
    # Setup monitor
    config = MonitorConfig(window_size=window_size, baseline_episodes=baseline_episodes)
    monitor = StrategyMonitor(config, n_actions=env.num_actions)
    
    # Collect data
    episodes = []
    cds_scores = []
    s_scores = []
    b_scores = []
    p_scores = []
    returns = []
    alerts = []
    
    for ep in range(n_episodes):
        opp = opp_list[0] if ep < change_point else opp_list[-1]
        
        if is_single:
            opp.get_action(np.zeros(env.state_shape), [0])
            if hasattr(opp, 'reset'):
                opp.reset()
        
        state = env.reset()
        agent.reset()
        states_ep = [state.observation.copy()]
        actions_ep = []
        rewards_ep = []
        
        while not state.is_terminal:
            if is_single or state.current_player == 0:
                action = agent.get_action(state.observation, state.legal_actions)
                actions_ep.append(action)
                rewards_ep.append(0.0)
            else:
                action = opp.get_action(state.observation, state.legal_actions)
            
            next_state, _ = env.step(action)
            state = next_state
            states_ep.append(state.observation.copy())
        
        ret = state.returns[0] if state.returns else 0.0
        if rewards_ep:
            rewards_ep[-1] = ret
        
        alert = monitor.update(states_ep, actions_ep, rewards_ep, ret)
        
        # Record data
        episodes.append(ep)
        returns.append(ret)
        
        if monitor.history:
            latest = monitor.history[-1]
            cds_scores.append(latest.composite_score)
            s_scores.append(latest.z_scores.get('S', 0))
            b_scores.append(latest.z_scores.get('B', 0))
            p_scores.append(latest.z_scores.get('P', 0))
        else:
            cds_scores.append(None)
            s_scores.append(None)
            b_scores.append(None)
            p_scores.append(None)
        
        if alert:
            alerts.append((ep, alert.composite_score, alert.level.value))
    
    return {
        'env_name': env_name,
        'episodes': episodes,
        'cds': cds_scores,
        's': s_scores,
        'b': b_scores,
        'p': p_scores,
        'returns': returns,
        'alerts': alerts,
        'change_point': change_point,
        'baseline_episodes': baseline_episodes,
    }


def plot_cds_timeseries(data: Dict, output_path: Path) -> None:
    """Plot CDS evolution over time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    eps = data['episodes']
    cds = [c if c is not None else np.nan for c in data['cds']]
    returns = data['returns']
    change_point = data['change_point']
    baseline_end = data['baseline_episodes']
    
    # Top: CDS
    ax1 = axes[0]
    ax1.plot(eps, cds, 'b-', linewidth=1.5, label='CDS')
    ax1.axvline(x=change_point, color='red', linestyle='--', linewidth=2, label='Change Point')
    ax1.axvline(x=baseline_end, color='gray', linestyle=':', linewidth=1.5, label='Baseline End')
    ax1.axhline(y=2.0, color='orange', linestyle='-', alpha=0.5, label='Yellow Threshold')
    ax1.axhline(y=3.0, color='red', linestyle='-', alpha=0.5, label='Red Threshold')
    
    # Mark alerts
    for ep, score, level in data['alerts']:
        color = 'red' if level == 'red' else 'orange'
        ax1.scatter([ep], [score], c=color, s=100, zorder=5, marker='^')
    
    # Shade regions
    ax1.axvspan(0, baseline_end, alpha=0.1, color='blue', label='Baseline')
    ax1.axvspan(change_point, max(eps), alpha=0.1, color='red', label='Post-Change')
    
    ax1.set_ylabel('Composite Drift Score (CDS)')
    ax1.set_title(f"SMF-AAS Monitoring: {data['env_name'].upper()}")
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Episode returns
    ax2 = axes[1]
    # Smooth returns with rolling average
    window = 20
    smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
    smooth_eps = eps[window-1:]
    
    ax2.plot(eps, returns, 'gray', alpha=0.3, linewidth=0.5)
    ax2.plot(smooth_eps, smoothed, 'green', linewidth=2, label=f'Return (smoothed, w={window})')
    ax2.axvline(x=change_point, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=baseline_end, color='gray', linestyle=':', linewidth=1.5)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Return')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.savefig(output_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {output_path.stem}.pdf/png")


def plot_components(data: Dict, output_path: Path) -> None:
    """Plot individual component z-scores."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    eps = data['episodes']
    change_point = data['change_point']
    baseline_end = data['baseline_episodes']
    
    components = [
        ('s', 'S (State Shift)', 'blue'),
        ('b', 'B (Behavior Shift)', 'green'),
        ('p', 'P (Performance)', 'purple'),
    ]
    
    for ax, (key, label, color) in zip(axes, components):
        values = [v if v is not None else np.nan for v in data[key]]
        
        ax.plot(eps, values, color=color, linewidth=1.5, label=label)
        ax.axvline(x=change_point, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=baseline_end, color='gray', linestyle=':', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=2, color='orange', linestyle='-', alpha=0.3)
        ax.axhline(y=-2, color='orange', linestyle='-', alpha=0.3)
        
        ax.set_ylabel(f'z-score ({label})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Episode')
    axes[0].set_title(f"Component z-Scores: {data['env_name'].upper()}")
    
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.savefig(output_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {output_path.stem}.pdf/png")


def plot_multi_env_comparison(all_data: Dict, output_path: Path) -> None:
    """Plot CDS comparison across all environments."""
    n_envs = len(all_data)
    fig, axes = plt.subplots(n_envs, 1, figsize=(12, 3*n_envs), sharex=True)
    
    if n_envs == 1:
        axes = [axes]
    
    for ax, (env_name, data) in zip(axes, all_data.items()):
        eps = data['episodes']
        cds = [c if c is not None else np.nan for c in data['cds']]
        change_point = data['change_point']
        baseline_end = data['baseline_episodes']
        
        ax.plot(eps, cds, 'b-', linewidth=1.5)
        ax.axvline(x=change_point, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=baseline_end, color='gray', linestyle=':', linewidth=1)
        ax.axhline(y=3.0, color='red', linestyle='-', alpha=0.3)
        
        # Mark first detection
        for ep, score, level in data['alerts']:
            if ep >= change_point:
                ax.scatter([ep], [score], c='red', s=100, zorder=5, marker='^')
                ax.annotate(f'Detected\n(delay={ep-change_point})', 
                           xy=(ep, score), xytext=(ep+20, score+1),
                           fontsize=9, ha='left',
                           arrowprops=dict(arrowstyle='->', color='red'))
                break
        
        ax.set_ylabel('CDS')
        ax.set_title(env_name.upper(), loc='left', fontweight='bold')
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Episode')
    fig.suptitle('CDS Evolution Across Environments', fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.savefig(output_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {output_path.stem}.pdf/png")


def main():
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    envs = ['tictactoe', 'connectfour', 'kuhnpoker', 'maze']
    all_data = {}
    
    print("Running monitored experiments...")
    for env_name in envs:
        print(f"  {env_name}...", end=" ")
        data = run_monitored_experiment(env_name, seed=42)
        all_data[env_name] = data
        
        # Individual plots
        plot_cds_timeseries(data, output_dir / f'timeseries_{env_name}')
        plot_components(data, output_dir / f'components_{env_name}')
        print("done")
    
    # Combined plot
    print("\nGenerating combined plot...")
    plot_multi_env_comparison(all_data, output_dir / 'timeseries_all_envs')
    
    print("\n✓ All time-series figures generated!")


if __name__ == "__main__":
    main()
