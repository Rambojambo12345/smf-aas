#!/usr/bin/env python3
"""
Main Experiment: Detection Performance Evaluation

This script runs the primary experiments for the SMF-AAS paper:
1. Detection rate across all environments
2. Detection delay analysis  
3. False positive rates
4. Comparison with baseline methods
5. Sensitivity analysis for parameters
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smf_aas import StrategyMonitor, MonitorConfig, get_env, list_envs
from smf_aas.agents import TabularQLearning
from smf_aas.baselines import CUSUM, ADWIN, PerformanceOnly


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class ExperimentConfig:
    env_name: str
    n_episodes: int = 1500
    change_point: int = 900
    baseline_episodes: int = 200
    window_size: int = 50
    training_episodes: int = 1000
    seed: int = 42
    
    # CUSUM baseline configuration (feedback: "Cusum baseline config discrete states")
    cusum_threshold: float = 5.0
    cusum_drift: float = 0.5
    cusum_warmup: int = 50
    
    # SMF-AAS thresholds
    yellow_threshold: float = 2.0
    red_threshold: float = 3.0
    
    # Log transform for CDS (feedback: "log consideration for CDS score")
    use_log_cds: bool = False


@dataclass 
class ExperimentResult:
    env_name: str
    seed: int
    config: Dict[str, Any]  # Store full config for reproducibility
    
    # SMF-AAS results
    detected: bool
    detection_delay: Optional[int]
    false_positives: int
    true_positives: int  # Added for precision/recall
    total_alerts: int
    baseline_stats: Dict[str, Dict[str, float]]
    
    # Precision/Recall metrics (feedback: "consider recall/precision")
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Baseline method results
    cusum_detected: bool = False
    cusum_delay: Optional[int] = None
    cusum_fp: int = 0
    cusum_tp: int = 0
    
    adwin_detected: bool = False
    adwin_delay: Optional[int] = None
    adwin_fp: int = 0
    adwin_tp: int = 0
    
    perf_only_detected: bool = False
    perf_only_delay: Optional[int] = None
    perf_only_fp: int = 0
    perf_only_tp: int = 0


# =============================================================================
# Core Experiment Functions
# =============================================================================

def run_episode(env, agent, opponent) -> tuple:
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

    if env.num_players == 1 and hasattr(opponent, 'configure'):
        opponent.configure()
    
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


def compute_precision_recall(tp: int, fp: int, fn: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = None
    
    return precision, recall, f1


def run_single_experiment(config: ExperimentConfig) -> ExperimentResult:
     # Setup environment
    env = get_env(config.env_name)
    opponents = env.get_opponents(config.seed)
    opp_list = list(opponents.values())
    
    if len(opp_list) < 2:
        opp_list = opp_list * 2
    
    # Train agent
    agent = train_agent(env, opp_list[0], config.training_episodes, config.seed)
    
    # Setup SMF-AAS monitor
    monitor_config = MonitorConfig(
        window_size=config.window_size,
        baseline_episodes=config.baseline_episodes,
        yellow_threshold=config.yellow_threshold,
        red_threshold=config.red_threshold,
    )
    monitor = StrategyMonitor(monitor_config, n_actions=env.num_actions)
    
    # Setup baseline detectors with configurable CUSUM parameters
    # Feedback: "Cusum baseline config discrete states"
    cusum = CUSUM(
        threshold=config.cusum_threshold,
        drift=config.cusum_drift,
        warmup=config.cusum_warmup
    )
    adwin = ADWIN(delta=0.002)
    perf_only = PerformanceOnly(
        window_size=config.window_size,
        baseline_episodes=config.baseline_episodes
    )
    
    # Tracking variables
    first_detection = None
    false_positives = 0
    true_positives = 0
    
    cusum_first = None
    cusum_fp = 0
    cusum_tp = 0
    adwin_first = None
    adwin_fp = 0
    adwin_tp = 0
    perf_first = None
    perf_fp = 0
    perf_tp = 0
    
    # Run experiment
    for ep in range(config.n_episodes):
        opp_idx = 0 if ep < config.change_point else 1
        opp = opp_list[opp_idx]
        
        if env.num_players == 1 and hasattr(opp, 'configure'):
            opp.configure()
        
        agent.reset()
        states, actions, rewards, ret = run_episode(env, agent, opp)
        
        # Update SMF-AAS monitor
        alert = monitor.update(states, actions, rewards, ret)
        
        if alert:
            if ep < config.change_point and ep >= config.baseline_episodes:
                false_positives += 1
            elif ep >= config.change_point:
                true_positives += 1
                if first_detection is None:
                    first_detection = ep
        
        # Update CUSUM
        cusum_result = cusum.update(ret)
        if cusum_result and cusum_result.detected:
            if ep < config.change_point and ep >= config.baseline_episodes:
                cusum_fp += 1
            elif ep >= config.change_point:
                cusum_tp += 1
                if cusum_first is None:
                    cusum_first = ep
        
        # Update ADWIN
        adwin_result = adwin.update(ret)
        if adwin_result and adwin_result.detected:
            if ep < config.change_point and ep >= config.baseline_episodes:
                adwin_fp += 1
            elif ep >= config.change_point:
                adwin_tp += 1
                if adwin_first is None:
                    adwin_first = ep
        
        # Update Performance-Only
        perf_result = perf_only.update(ret)
        if perf_result and perf_result.detected:
            if ep < config.change_point and ep >= config.baseline_episodes:
                perf_fp += 1
            elif ep >= config.change_point:
                perf_tp += 1
                if perf_first is None:
                    perf_first = ep
    
    # Compute metrics
    detected = first_detection is not None
    delay = first_detection - config.change_point if detected else None
    
    # For this experiment: 1 change occurs, so max TP=1 for "detected", FN=0 if detected, FN=1 if not
    fn = 0 if detected else 1
    precision, recall, f1 = compute_precision_recall(1 if detected else 0, false_positives, fn)
    
    cusum_detected = cusum_first is not None
    cusum_delay = cusum_first - config.change_point if cusum_detected else None
    
    adwin_detected = adwin_first is not None
    adwin_delay = adwin_first - config.change_point if adwin_detected else None
    
    perf_detected = perf_first is not None
    perf_delay = perf_first - config.change_point if perf_detected else None
    
    return ExperimentResult(
        env_name=config.env_name,
        seed=config.seed,
        config=asdict(config),
        detected=detected,
        detection_delay=delay,
        false_positives=false_positives,
        true_positives=true_positives,
        total_alerts=len(monitor.alerts),
        baseline_stats={
            k: {"mean": v[0], "std": v[1]}
            for k, v in monitor.baseline_stats.items()
        },
        precision=precision,
        recall=recall,
        f1_score=f1,
        cusum_detected=cusum_detected,
        cusum_delay=cusum_delay,
        cusum_fp=cusum_fp,
        cusum_tp=cusum_tp,
        adwin_detected=adwin_detected,
        adwin_delay=adwin_delay,
        adwin_fp=adwin_fp,
        adwin_tp=adwin_tp,
        perf_only_detected=perf_detected,
        perf_only_delay=perf_delay,
        perf_only_fp=perf_fp,
        perf_only_tp=perf_tp,
    )


# =============================================================================
# Multi-seed and Aggregation Functions
# =============================================================================

def aggregate_method(detected_list, delays, fps, tps) -> Dict[str, Any]:
    valid_delays = [d for d in delays if d is not None]
    
    # Compute precision/recall across all runs
    total_tp = sum(1 for d in detected_list if d)
    total_fp = sum(fps)
    total_fn = sum(1 for d in detected_list if not d)
    
    precision, recall, f1 = compute_precision_recall(total_tp, total_fp, total_fn)
    
    return {
        "detection_rate": float(np.mean(detected_list)),
        "mean_delay": float(np.mean(valid_delays)) if valid_delays else None,
        "std_delay": float(np.std(valid_delays)) if len(valid_delays) > 1 else None,
        "mean_fp": float(np.mean(fps)),
        "std_fp": float(np.std(fps)) if len(fps) > 1 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def run_multi_seed(
    env_name: str,
    n_seeds: int = 10,
    seed_start: int = 42,
    seed_step: int = 1000,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    results = []
    seeds_used = []
    
    for i in range(n_seeds):
        seed = seed_start + i * seed_step
        seeds_used.append(seed)
        config = ExperimentConfig(env_name=env_name, seed=seed, **kwargs)
        
        if verbose:
            print(f"  Seed {i+1}/{n_seeds} (seed={seed})...", end=" ", flush=True)
        
        result = run_single_experiment(config)
        results.append(result)
        
        if verbose:
            status = "✓" if result.detected else "✗"
            delay_str = str(result.detection_delay) if result.detection_delay is not None else "N/A"
            print(f"{status} delay={delay_str}, FP={result.false_positives}")
    
    # Aggregate results
    summary = {
        "smf_aas": aggregate_method(
            [r.detected for r in results],
            [r.detection_delay for r in results],
            [r.false_positives for r in results],
            [r.true_positives for r in results]
        ),
        "cusum": aggregate_method(
            [r.cusum_detected for r in results],
            [r.cusum_delay for r in results],
            [r.cusum_fp for r in results],
            [r.cusum_tp for r in results]
        ),
        "adwin": aggregate_method(
            [r.adwin_detected for r in results],
            [r.adwin_delay for r in results],
            [r.adwin_fp for r in results],
            [r.adwin_tp for r in results]
        ),
        "perf_only": aggregate_method(
            [r.perf_only_detected for r in results],
            [r.perf_only_delay for r in results],
            [r.perf_only_fp for r in results],
            [r.perf_only_tp for r in results]
        ),
    }
    
    return {
        "env_name": env_name,
        "n_seeds": n_seeds,
        "seeds": seeds_used,
        "config": {
            "n_episodes": kwargs.get('n_episodes', 1500),
            "change_point": kwargs.get('change_point', 900),
            "baseline_episodes": kwargs.get('baseline_episodes', 200),
            "window_size": kwargs.get('window_size', 50),
            "training_episodes": kwargs.get('training_episodes', 1000),
            "cusum_threshold": kwargs.get('cusum_threshold', 5.0),
            "cusum_drift": kwargs.get('cusum_drift', 0.5),
        },
        "individual_results": [asdict(r) for r in results],
        "summary": summary,
    }


# =============================================================================
# Sensitivity Analysis (Feedback: "Sensitivity to parameters")
# =============================================================================

def run_sensitivity_analysis(
    env_name: str,
    parameter: str,
    n_seeds: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
      
    PARAMETER_RANGES = {
        "window_size": [20, 30, 50, 75, 100],  
        "training_episodes": [500, 750, 1000, 1500, 2000],  
        "baseline_episodes": [100, 150, 200, 300, 400],
        "cusum_threshold": [3.0, 4.0, 5.0, 6.0, 8.0],  
        "cusum_drift": [0.25, 0.5, 0.75, 1.0],
        "yellow_threshold": [1.5, 2.0, 2.5, 3.0],
        "red_threshold": [2.5, 3.0, 3.5, 4.0],
    }
    
    if parameter not in PARAMETER_RANGES:
        raise ValueError(f"Unknown parameter: {parameter}. Available: {list(PARAMETER_RANGES.keys())}")
    
    values = PARAMETER_RANGES[parameter]
    results = {}
    
    print(f"\nSensitivity Analysis: {parameter}")
    print("=" * 60)
    
    for value in values:
        print(f"\n{parameter}={value}")
        print("-" * 40)
        
        kwargs = {parameter: value}
        
        # Adjust baseline_episodes if window_size changes
        if parameter == "window_size":
            kwargs["baseline_episodes"] = max(200, value * 4)
        
        result = run_multi_seed(
            env_name,
            n_seeds=n_seeds,
            verbose=verbose,
            **kwargs
        )
        
        results[str(value)] = result
    
    return {
        "parameter": parameter,
        "values_tested": values,
        "env_name": env_name,
        "n_seeds": n_seeds,
        "results": results,
    }


# =============================================================================
# Result Merging (for chunked running)
# =============================================================================

def merge_results(output_dir: Path) -> Dict[str, Any]:
    partial_files = list(output_dir.glob("partial_*.json"))
    
    if not partial_files:
        print(f"No partial result files found in {output_dir}")
        return {}
    
    print(f"Found {len(partial_files)} partial result files:")
    for f in sorted(partial_files):
        print(f"  - {f.name}")
    
    all_results = {}
    
    for filepath in partial_files:
        with open(filepath) as f:
            data = json.load(f)
        
        # Handle both single-env and multi-env files
        if isinstance(data, dict):
            for env_name, env_data in data.items():
                if env_name not in all_results:
                    all_results[env_name] = env_data
                else:
                    # Merge results from same environment
                    existing = all_results[env_name]
                    existing["individual_results"].extend(env_data["individual_results"])
                    existing["seeds"].extend(env_data.get("seeds", []))
                    existing["n_seeds"] = len(existing["individual_results"])
                    
                    # Recompute summary
                    results = existing["individual_results"]
                    existing["summary"] = {
                        "smf_aas": aggregate_method(
                            [r["detected"] for r in results],
                            [r["detection_delay"] for r in results],
                            [r["false_positives"] for r in results],
                            [r.get("true_positives", 1 if r["detected"] else 0) for r in results]
                        ),
                        "cusum": aggregate_method(
                            [r["cusum_detected"] for r in results],
                            [r["cusum_delay"] for r in results],
                            [r["cusum_fp"] for r in results],
                            [r.get("cusum_tp", 1 if r["cusum_detected"] else 0) for r in results]
                        ),
                        "adwin": aggregate_method(
                            [r["adwin_detected"] for r in results],
                            [r["adwin_delay"] for r in results],
                            [r["adwin_fp"] for r in results],
                            [r.get("adwin_tp", 1 if r["adwin_detected"] else 0) for r in results]
                        ),
                        "perf_only": aggregate_method(
                            [r["perf_only_detected"] for r in results],
                            [r["perf_only_delay"] for r in results],
                            [r["perf_only_fp"] for r in results],
                            [r.get("perf_only_tp", 1 if r["perf_only_detected"] else 0) for r in results]
                        ),
                    }
    
    # Save merged results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_file = output_dir / f"merged_results_{timestamp}.json"
    
    with open(merged_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nMerged results saved to: {merged_file}")
    
    return all_results


# =============================================================================
# Output Functions
# =============================================================================

def print_summary_table(all_results: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    
    # SMF-AAS results
    print("\nSMF-AAS Framework:")
    print("-" * 90)
    print(f"{'Environment':<15} {'Det.Rate':<10} {'Delay':<15} {'FP':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 90)
    
    for env_name, data in sorted(all_results.items()):
        s = data["summary"]["smf_aas"]
        rate = f"{s['detection_rate']:.0%}"
        if s["mean_delay"] is not None:
            delay = f"{s['mean_delay']:.1f}"
        else:
            delay = "N/A"
        fp = f"{s['mean_fp']:.1f}"
        prec = f"{s['precision']:.2f}" if s.get('precision') is not None else "N/A"
        rec = f"{s['recall']:.2f}" if s.get('recall') is not None else "N/A"
        f1 = f"{s['f1_score']:.2f}" if s.get('f1_score') is not None else "N/A"
        print(f"{env_name:<15} {rate:<10} {delay:<15} {fp:<10} {prec:<10} {rec:<10} {f1:<10}")
    
    # Method Comparison
    print("\n\nMethod Comparison (averaged across environments):")
    print("-" * 90)
    print(f"{'Method':<15} {'Det.Rate':<10} {'Mean Delay':<15} {'Mean FP':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 90)
    
    methods = ["smf_aas", "cusum", "adwin", "perf_only"]
    method_names = ["SMF-AAS", "CUSUM", "ADWIN", "Perf-Only"]
    
    for method, name in zip(methods, method_names):
        rates, delays, fps, precs, recs = [], [], [], [], []
        
        for data in all_results.values():
            s = data["summary"][method]
            rates.append(s["detection_rate"])
            if s["mean_delay"] is not None:
                delays.append(s["mean_delay"])
            fps.append(s["mean_fp"])
            if s.get("precision") is not None:
                precs.append(s["precision"])
            if s.get("recall") is not None:
                recs.append(s["recall"])
        
        rate = f"{np.mean(rates):.0%}"
        delay = f"{np.mean(delays):.1f}" if delays else "N/A"
        fp = f"{np.mean(fps):.1f}"
        prec = f"{np.mean(precs):.2f}" if precs else "N/A"
        rec = f"{np.mean(recs):.2f}" if recs else "N/A"
        
        print(f"{name:<15} {rate:<10} {delay:<15} {fp:<10} {prec:<10} {rec:<10}")


def save_results(results: Dict[str, Any], output_dir: Path, prefix: str = "experiment") -> Path:
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{prefix}_results_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return output_file


# =============================================================================
# Mn entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run SMF-AAS detection experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CHUNKED RUNNING:
  python run_main_experiment.py --env tictactoe --seeds 20
  python run_main_experiment.py --env connectfour --seeds 20
  python run_main_experiment.py --merge

SENSITIVITY ANALYSIS:
  python run_main_experiment.py --env tictactoe --sensitivity window_size
  python run_main_experiment.py --env tictactoe --sensitivity training_episodes

FULL RUN:
  python run_main_experiment.py --all --seeds 20
        """
    )
    
    # Environment selection
    parser.add_argument("--env", type=str, help="Single environment to test")
    parser.add_argument("--envs", type=str, nargs='+', help="Multiple environments")
    parser.add_argument("--all", action="store_true", help="Run all environments")
    parser.add_argument("--internal-only", action="store_true", help="Internal environments only")
    parser.add_argument("--external-only", action="store_true", help="External environments only")
    
    # Seed configuration (Feedback: "Increase seeds and play with this")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds (default: 10)")
    parser.add_argument("--seed-start", type=int, default=42, help="Starting seed (default: 42)")
    parser.add_argument("--seed-step", type=int, default=1000, help="Seed increment (default: 1000)")
    
    # Core parameters (Feedback: various parameter tweaking)
    parser.add_argument("--episodes", type=int, default=1500, help="Total episodes (default: 1500)")
    parser.add_argument("--change-point", type=int, default=900, help="Change episode (default: 900)")
    parser.add_argument("--window-size", type=int, default=50, help="Window size (default: 50)")
    parser.add_argument("--baseline-episodes", type=int, default=200, help="Baseline period (default: 200)")
    parser.add_argument("--training-episodes", type=int, default=1000, help="Training episodes (default: 1000)")
    
    # CUSUM configuration (Feedback: "Cusum baseline config discrete states")
    parser.add_argument("--cusum-threshold", type=float, default=5.0, help="CUSUM threshold (default: 5.0)")
    parser.add_argument("--cusum-drift", type=float, default=0.5, help="CUSUM drift (default: 0.5)")
    
    # Sensitivity analysis (Feedback: "Sensitivity to parameters")
    parser.add_argument("--sensitivity", type=str, 
                       choices=["window_size", "training_episodes", "baseline_episodes", 
                               "cusum_threshold", "cusum_drift", "yellow_threshold", "red_threshold", "all"],
                       help="Run sensitivity analysis for parameter")
    
    # Output and merging
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--merge", action="store_true", help="Merge partial results")
    parser.add_argument("--save-partial", action="store_true", default=True,
                       help="Save results per environment (for chunked running)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Handle merge operation
    if args.merge:
        all_results = merge_results(output_dir)
        if all_results:
            print_summary_table(all_results)
        return
    
    # Handle sensitivity analysis
    if args.sensitivity:
        if not args.env:
            print("Error: --sensitivity requires --env")
            return
        
        if args.sensitivity == "all":
            all_sensitivity = {}
            for param in ["window_size", "training_episodes", "cusum_threshold"]:
                result = run_sensitivity_analysis(args.env, param, n_seeds=args.seeds)
                all_sensitivity[param] = result
                save_results(result, output_dir, prefix=f"sensitivity_{param}")
        else:
            result = run_sensitivity_analysis(args.env, args.sensitivity, n_seeds=args.seeds)
            save_results(result, output_dir, prefix=f"sensitivity_{args.sensitivity}")
        return
    
    # Determine environments
    internal_envs = ["tictactoe", "connectfour", "kuhnpoker", "maze"]
    external_envs = ["tictactoe-pz", "connectfour-pz", "cartpole-gym"]
    
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
        envs = internal_envs
    
    # Print configuration
    print("=" * 70)
    print("SMF-AAS Experiment Runner")
    print("=" * 70)
    print(f"Environments: {envs}")
    print(f"Seeds: {args.seeds} (start={args.seed_start}, step={args.seed_step})")
    print(f"Episodes: {args.episodes}, Change point: {args.change_point}")
    print(f"Window size: {args.window_size}, Baseline: {args.baseline_episodes}")
    print(f"Training: {args.training_episodes} episodes")
    print(f"CUSUM: threshold={args.cusum_threshold}, drift={args.cusum_drift}")
    print()
    
    # Build kwargs
    exp_kwargs = {
        "n_episodes": args.episodes,
        "change_point": args.change_point,
        "window_size": args.window_size,
        "baseline_episodes": args.baseline_episodes,
        "training_episodes": args.training_episodes,
        "cusum_threshold": args.cusum_threshold,
        "cusum_drift": args.cusum_drift,
    }
    
    all_results = {}
    
    for env_name in envs:
        print(f"\n{'='*60}")
        print(f"Environment: {env_name}")
        print(f"{'='*60}")
        
        try:
            result = run_multi_seed(
                env_name,
                n_seeds=args.seeds,
                seed_start=args.seed_start,
                seed_step=args.seed_step,
                **exp_kwargs
            )
            all_results[env_name] = result
            
            # Save partial results for chunked running
            if args.save_partial:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                partial_file = output_dir / f"partial_{env_name}_{timestamp}.json"
                with open(partial_file, "w") as f:
                    json.dump({env_name: result}, f, indent=2)
                print(f"\n  Saved: {partial_file}")
                
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Print and save final results
    if all_results:
        print_summary_table(all_results)
        output_file = save_results(all_results, output_dir)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
