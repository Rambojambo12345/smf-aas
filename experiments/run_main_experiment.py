#!/usr/bin/env python3
"""
Main Experiment: Detection Performance Evaluation

This script runs the primary experiments for the SMF-AAS paper:
1. Detection rate across all environments
2. Detection delay analysis
3. False positive rates
4. Comparison with baseline methods

Usage
-----
    python experiments/run_main_experiment.py --seeds 10
    python experiments/run_main_experiment.py --env tictactoe --seeds 5
    python experiments/run_main_experiment.py --all --seeds 10 --output results/

Results are saved as JSON and can be used to generate paper figures.
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smf_aas import StrategyMonitor, MonitorConfig, get_env, list_envs
from smf_aas.agents import TabularQLearning
from smf_aas.baselines import CUSUM, ADWIN, PerformanceOnly


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    env_name: str
    n_episodes: int = 1500
    change_point: int = 900
    baseline_episodes: int = 200
    window_size: int = 50
    training_episodes: int = 1000
    seed: int = 42


@dataclass 
class ExperimentResult:
    """Results from a single experiment run."""
    env_name: str
    seed: int
    detected: bool
    detection_delay: Optional[int]
    false_positives: int
    total_alerts: int
    baseline_stats: Dict[str, Dict[str, float]]
    
    # Baseline method results
    cusum_detected: bool = False
    cusum_delay: Optional[int] = None
    cusum_fp: int = 0
    
    adwin_detected: bool = False
    adwin_delay: Optional[int] = None
    adwin_fp: int = 0
    
    perf_only_detected: bool = False
    perf_only_delay: Optional[int] = None
    perf_only_fp: int = 0


def run_episode(env, agent, opponent) -> tuple:
    """Run single episode and collect monitoring data.
    
    Returns
    -------
    tuple
        (states, actions, rewards, episode_return)
    """
    states, actions, rewards = [], [], []
    state = env.reset()
    states.append(state.observation.copy())
    
    is_single_player = env.num_players == 1
    
    while not state.is_terminal:
        if is_single_player or state.current_player == 0:
            action = agent.get_action(state.observation, state.legal_actions)
            actions.append(action)
            rewards.append(0.0)
        else:
            action = opponent.get_action(state.observation, state.legal_actions)
        
        state, done = env.step(action)
        states.append(state.observation.copy())
        
        # Update agent if it's a learning agent
        if is_single_player or state.current_player == 1 or done:
            if hasattr(agent, 'update') and actions:
                reward = state.returns[0] if done and state.returns else 0.0
                agent.update(
                    reward,
                    state.observation,
                    state.legal_actions if not done else [],
                    done
                )
    
    episode_return = state.returns[0] if state.returns else 0.0
    if rewards:
        rewards[-1] = episode_return
    
    return states, actions, rewards, episode_return


def train_agent(env, opponent, n_episodes: int, seed: int) -> TabularQLearning:
    """Train Q-learning agent against opponent."""
    agent = TabularQLearning(
        n_actions=env.num_actions,
        learning_rate=0.1,
        discount=0.99,
        epsilon=0.15,
        seed=seed
    )
    
    for _ in range(n_episodes):
        agent.reset()
        run_episode(env, agent, opponent)
    
    # Reduce exploration after training
    agent.epsilon = 0.05
    return agent


def run_single_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run single experiment with given configuration."""
    # Setup environment
    env = get_env(config.env_name)
    opponents = env.get_opponents(config.seed)
    opp_list = list(opponents.values())
    
    if len(opp_list) < 2:
        # Single-player or single opponent: duplicate for "change"
        opp_list = opp_list * 2
    
    # Train agent against first opponent
    agent = train_agent(env, opp_list[0], config.training_episodes, config.seed)
    
    # Setup monitor
    monitor_config = MonitorConfig(
        window_size=config.window_size,
        baseline_episodes=config.baseline_episodes,
        yellow_threshold=2.0,
        red_threshold=3.0,
    )
    monitor = StrategyMonitor(monitor_config, n_actions=env.num_actions)
    
    # Setup baseline detectors
    cusum = CUSUM(threshold=5.0, warmup=50)
    adwin = ADWIN(delta=0.002)
    perf_only = PerformanceOnly(
        window_size=config.window_size,
        baseline_episodes=config.baseline_episodes
    )
    
    # Run experiment
    first_detection = None
    false_positives = 0
    
    cusum_first = None
    cusum_fp = 0
    adwin_first = None
    adwin_fp = 0
    perf_first = None
    perf_fp = 0
    
    for ep in range(config.n_episodes):
        # Switch opponent at change_point
        opp_idx = 0 if ep < config.change_point else 1
        opp = opp_list[opp_idx]
        
        # Run episode
        agent.reset()
        states, actions, rewards, ret = run_episode(env, agent, opp)
        
        # Update SMF-AAS monitor
        alert = monitor.update(states, actions, rewards, ret)
        
        if alert:
            if ep < config.change_point and ep >= config.baseline_episodes:
                false_positives += 1
            elif ep >= config.change_point and first_detection is None:
                first_detection = ep
        
        # Update baseline methods
        cusum_result = cusum.update(ret)
        if cusum_result and cusum_result.detected:
            if ep < config.change_point and ep >= config.baseline_episodes:
                cusum_fp += 1
            elif ep >= config.change_point and cusum_first is None:
                cusum_first = ep
        
        adwin_result = adwin.update(ret)
        if adwin_result and adwin_result.detected:
            if ep < config.change_point and ep >= config.baseline_episodes:
                adwin_fp += 1
            elif ep >= config.change_point and adwin_first is None:
                adwin_first = ep
        
        perf_result = perf_only.update(ret)
        if perf_result and perf_result.detected:
            if ep < config.change_point and ep >= config.baseline_episodes:
                perf_fp += 1
            elif ep >= config.change_point and perf_first is None:
                perf_first = ep
    
    # Compute results
    detected = first_detection is not None
    delay = first_detection - config.change_point if detected else None
    
    cusum_detected = cusum_first is not None
    cusum_delay = cusum_first - config.change_point if cusum_detected else None
    
    adwin_detected = adwin_first is not None
    adwin_delay = adwin_first - config.change_point if adwin_detected else None
    
    perf_detected = perf_first is not None
    perf_delay = perf_first - config.change_point if perf_detected else None
    
    return ExperimentResult(
        env_name=config.env_name,
        seed=config.seed,
        detected=detected,
        detection_delay=delay,
        false_positives=false_positives,
        total_alerts=len(monitor.alerts),
        baseline_stats={
            k: {"mean": v[0], "std": v[1]}
            for k, v in monitor.baseline_stats.items()
        },
        cusum_detected=cusum_detected,
        cusum_delay=cusum_delay,
        cusum_fp=cusum_fp,
        adwin_detected=adwin_detected,
        adwin_delay=adwin_delay,
        adwin_fp=adwin_fp,
        perf_only_detected=perf_detected,
        perf_only_delay=perf_delay,
        perf_only_fp=perf_fp,
    )


def run_multi_seed(
    env_name: str,
    n_seeds: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """Run experiment across multiple seeds."""
    results = []
    
    for i in range(n_seeds):
        seed = 42 + i * 1000
        config = ExperimentConfig(env_name=env_name, seed=seed, **kwargs)
        
        print(f"  Seed {i+1}/{n_seeds} (seed={seed})...", end=" ", flush=True)
        result = run_single_experiment(config)
        results.append(result)
        
        status = "✓" if result.detected else "✗"
        delay_str = str(result.detection_delay) if result.detection_delay is not None else "N/A"
        print(f"{status} delay={delay_str}, FP={result.false_positives}")
    
    # Aggregate results
    def aggregate_method(detected_list, delays, fps):
        valid_delays = [d for d in delays if d is not None]
        return {
            "detection_rate": float(np.mean(detected_list)),
            "mean_delay": float(np.mean(valid_delays)) if valid_delays else None,
            "std_delay": float(np.std(valid_delays)) if len(valid_delays) > 1 else None,
            "mean_fp": float(np.mean(fps)),
            "std_fp": float(np.std(fps)) if len(fps) > 1 else 0.0,
        }
    
    summary = {
        "smf_aas": aggregate_method(
            [r.detected for r in results],
            [r.detection_delay for r in results],
            [r.false_positives for r in results]
        ),
        "cusum": aggregate_method(
            [r.cusum_detected for r in results],
            [r.cusum_delay for r in results],
            [r.cusum_fp for r in results]
        ),
        "adwin": aggregate_method(
            [r.adwin_detected for r in results],
            [r.adwin_delay for r in results],
            [r.adwin_fp for r in results]
        ),
        "perf_only": aggregate_method(
            [r.perf_only_detected for r in results],
            [r.perf_only_delay for r in results],
            [r.perf_only_fp for r in results]
        ),
    }
    
    return {
        "env_name": env_name,
        "n_seeds": n_seeds,
        "config": {
            "n_episodes": kwargs.get('n_episodes', 1500),
            "change_point": kwargs.get('change_point', 900),
            "baseline_episodes": kwargs.get('baseline_episodes', 200),
            "window_size": kwargs.get('window_size', 50),
            "training_episodes": kwargs.get('training_episodes', 1000),
        },
        "individual_results": [asdict(r) for r in results],
        "summary": summary,
    }


def print_summary_table(all_results: Dict[str, Any]) -> None:
    """Print formatted summary table."""
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    
    # SMF-AAS results
    print("\nSMF-AAS Framework:")
    print("-" * 70)
    print(f"{'Environment':<15} {'Det.Rate':<12} {'Delay':<18} {'FP':<12}")
    print("-" * 70)
    
    for env_name, data in all_results.items():
        s = data["summary"]["smf_aas"]
        rate = f"{s['detection_rate']:.0%}"
        if s["mean_delay"] is not None:
            std_str = f"{s['std_delay']:.1f}" if s["std_delay"] is not None else "N/A"
            delay = f"{s['mean_delay']:.1f} ± {std_str}"
        else:
            delay = "N/A"
        fp = f"{s['mean_fp']:.1f} ± {s['std_fp']:.1f}"
        print(f"{env_name:<15} {rate:<12} {delay:<18} {fp:<12}")
    
    # Comparison table
    print("\n\nMethod Comparison (averaged across environments):")
    print("-" * 70)
    print(f"{'Method':<15} {'Det.Rate':<12} {'Mean Delay':<18} {'Mean FP':<12}")
    print("-" * 70)
    
    methods = ["smf_aas", "cusum", "adwin", "perf_only"]
    method_names = ["SMF-AAS", "CUSUM", "ADWIN", "Perf-Only"]
    
    for method, name in zip(methods, method_names):
        rates = []
        delays = []
        fps = []
        
        for data in all_results.values():
            s = data["summary"][method]
            rates.append(s["detection_rate"])
            if s["mean_delay"] is not None:
                delays.append(s["mean_delay"])
            fps.append(s["mean_fp"])
        
        rate = f"{np.mean(rates):.0%}"
        delay = f"{np.mean(delays):.1f}" if delays else "N/A"
        fp = f"{np.mean(fps):.1f}"
        
        print(f"{name:<15} {rate:<12} {delay:<18} {fp:<12}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SMF-AAS detection experiments"
    )
    parser.add_argument(
        "--env", type=str, default=None,
        help="Specific environment to test"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run on all environments"
    )
    parser.add_argument(
        "--envs", type=str, nargs='+',
        help="Multiple environments to test (e.g., --envs tictactoe connectfour)"
    )
    parser.add_argument(
        "--seeds", type=int, default=10,
        help="Number of random seeds (default: 10)"
    )
    parser.add_argument(
        "--episodes", type=int, default=1500,
        help="Total episodes per run (default: 1500)"
    )
    parser.add_argument(
        "--change-point", type=int, default=900,
        help="Episode where change occurs (default: 900)"
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory (default: results)"
    )
    parser.add_argument(
        "--internal-only", action="store_true",
        help="Run only internal environments (no external packages needed)"
    )
    parser.add_argument(
        "--external-only", action="store_true",
        help="Run only external environments (requires pettingzoo, gymnasium)"
    )
    
    args = parser.parse_args()
    
    # Define environment groups
    internal_envs = ["tictactoe", "connectfour", "kuhnpoker", "maze"]
    external_envs = ["tictactoe-pz", "connectfour-pz", "cartpole-gym"]
    
    # Determine environments to test
    if args.all:
        envs = list_envs()
    elif args.envs:
        envs = args.envs
    elif args.env:
        envs = [args.env]
    elif args.internal_only:
        envs = internal_envs
    elif args.external_only:
        envs = external_envs
    else:
        # Default: internal environments
        envs = internal_envs
    
    print("=" * 70)
    print("SMF-AAS Experiment Runner")
    print("=" * 70)
    print(f"Environments: {envs}")
    print(f"Seeds: {args.seeds}")
    print(f"Episodes: {args.episodes}")
    print(f"Change point: {args.change_point}")
    print()
    
    all_results = {}
    
    for env_name in envs:
        print(f"\n{'='*60}")
        print(f"Environment: {env_name}")
        print(f"{'='*60}")
        
        try:
            result = run_multi_seed(
                env_name,
                n_seeds=args.seeds,
                n_episodes=args.episodes,
                change_point=args.change_point,
            )
            all_results[env_name] = result
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    if all_results:
        print_summary_table(all_results)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"experiment_results_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
