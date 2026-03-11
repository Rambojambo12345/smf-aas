#!/usr/bin/env python3
"""
Additional Baselines Experiment Runner

This script runs ONLY the new baseline methods!!!!!!!!! (DDM, EDDM, KSWIN, Page-Hinkley)


The new baselines being tested:
- DDM (Drift Detection Method) - EDDM (Early Drift Detection Method) 
- KSWIN (Kolmogorov-Smirnov Windowing) 
- Page-Hinkley Test  
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smf_aas import StrategyMonitor, MonitorConfig, get_env, list_envs
from smf_aas.agents import TabularQLearning
from smf_aas.baselines import DDM, EDDM, KSWIN, PageHinkley


@dataclass
class ExperimentConfig:
    """Configuration matching main experiments."""
    env_name: str
    n_episodes: int = 1500
    change_point: int = 900
    baseline_episodes: int = 200
    window_size: int = 50
    training_episodes: int = 1000
    seed: int = 42


@dataclass 
class BaselineResult:
    """Results from additional baseline methods."""
    env_name: str
    seed: int
    
    # DDM results
    ddm_detected: bool = False
    ddm_delay: Optional[int] = None
    ddm_fp: int = 0
    
    # EDDM results
    eddm_detected: bool = False
    eddm_delay: Optional[int] = None
    eddm_fp: int = 0
    
    # KSWIN results
    kswin_detected: bool = False
    kswin_delay: Optional[int] = None
    kswin_fp: int = 0
    
    # Page-Hinkley results
    ph_detected: bool = False
    ph_delay: Optional[int] = None
    ph_fp: int = 0


def run_episode(env, agent, opponent) -> tuple:
    """Run single episode and collect data."""
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
    """Train Q-learning agent (same as main experiments)."""
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
    
    agent.epsilon = 0.05
    return agent


def run_single_experiment(config: ExperimentConfig) -> BaselineResult:
    """Run experiment with additional baselines."""
    # Setup environment (same as main experiments)
    env = get_env(config.env_name)
    opponents = env.get_opponents(config.seed)
    opp_list = list(opponents.values())
    
    if len(opp_list) < 2:
        opp_list = opp_list * 2
    
    # Train agent
    agent = train_agent(env, opp_list[0], config.training_episodes, config.seed)
    
    # Setup new baseline detectors with default parameters
    ddm = DDM(warning_level=2.0, drift_level=3.0, min_instances=30)
    eddm = EDDM(warning_level=0.95, drift_level=0.90, min_instances=30)
    kswin = KSWIN(alpha=0.005, window_size=100, stat_size=30)
    ph = PageHinkley(threshold=50.0, alpha=0.005, min_instances=50)
    
    # Tracking variables
    ddm_first, ddm_fp = None, 0
    eddm_first, eddm_fp = None, 0
    kswin_first, kswin_fp = None, 0
    ph_first, ph_fp = None, 0
    
    # Run experiment
    for ep in range(config.n_episodes):
        opp_idx = 0 if ep < config.change_point else 1
        opp = opp_list[opp_idx]
        
        if env.num_players == 1 and hasattr(opp, 'configure'):
            opp.configure()
        
        agent.reset()
        states, actions, rewards, ret = run_episode(env, agent, opp)
        
        # Update DDM
        ddm_result = ddm.update(ret)
        if ddm_result and ddm_result.detected:
            if ep < config.change_point and ep >= config.baseline_episodes:
                ddm_fp += 1
            elif ep >= config.change_point and ddm_first is None:
                ddm_first = ep
        
        # Update EDDM
        eddm_result = eddm.update(ret)
        if eddm_result and eddm_result.detected:
            if ep < config.change_point and ep >= config.baseline_episodes:
                eddm_fp += 1
            elif ep >= config.change_point and eddm_first is None:
                eddm_first = ep
        
        # Update KSWIN
        kswin_result = kswin.update(ret)
        if kswin_result and kswin_result.detected:
            if ep < config.change_point and ep >= config.baseline_episodes:
                kswin_fp += 1
            elif ep >= config.change_point and kswin_first is None:
                kswin_first = ep
        
        # Update Page-Hinkley
        ph_result = ph.update(ret)
        if ph_result and ph_result.detected:
            if ep < config.change_point and ep >= config.baseline_episodes:
                ph_fp += 1
            elif ep >= config.change_point and ph_first is None:
                ph_first = ep
    
    return BaselineResult(
        env_name=config.env_name,
        seed=config.seed,
        ddm_detected=ddm_first is not None,
        ddm_delay=ddm_first - config.change_point if ddm_first else None,
        ddm_fp=ddm_fp,
        eddm_detected=eddm_first is not None,
        eddm_delay=eddm_first - config.change_point if eddm_first else None,
        eddm_fp=eddm_fp,
        kswin_detected=kswin_first is not None,
        kswin_delay=kswin_first - config.change_point if kswin_first else None,
        kswin_fp=kswin_fp,
        ph_detected=ph_first is not None,
        ph_delay=ph_first - config.change_point if ph_first else None,
        ph_fp=ph_fp,
    )


def compute_precision_recall(tp: int, fp: int, fn: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = None
    return precision, recall, f1


def aggregate_method(detected_list, delays, fps) -> Dict[str, Any]:
    """Aggregate results across seeds."""
    valid_delays = [d for d in delays if d is not None]
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


def run_multi_seed(env_name: str, n_seeds: int = 20, seed_start: int = 42, seed_step: int = 1000) -> Dict[str, Any]:
    """Run experiment across multiple seeds."""
    results = []
    seeds_used = []
    
    for i in range(n_seeds):
        seed = seed_start + i * seed_step
        seeds_used.append(seed)
        config = ExperimentConfig(env_name=env_name, seed=seed)
        
        print(f"  Seed {i+1}/{n_seeds} (seed={seed})...", end=" ", flush=True)
        result = run_single_experiment(config)
        results.append(result)
        
        # Show quick status
        detected = [result.ddm_detected, result.eddm_detected, result.kswin_detected, result.ph_detected]
        det_count = sum(detected)
        print(f"detected: {det_count}/4 methods")
    
    # Aggregate
    summary = {
        "ddm": aggregate_method(
            [r.ddm_detected for r in results],
            [r.ddm_delay for r in results],
            [r.ddm_fp for r in results]
        ),
        "eddm": aggregate_method(
            [r.eddm_detected for r in results],
            [r.eddm_delay for r in results],
            [r.eddm_fp for r in results]
        ),
        "kswin": aggregate_method(
            [r.kswin_detected for r in results],
            [r.kswin_delay for r in results],
            [r.kswin_fp for r in results]
        ),
        "page_hinkley": aggregate_method(
            [r.ph_detected for r in results],
            [r.ph_delay for r in results],
            [r.ph_fp for r in results]
        ),
    }
    
    return {
        "env_name": env_name,
        "n_seeds": n_seeds,
        "seeds": seeds_used,
        "individual_results": [asdict(r) for r in results],
        "summary": summary,
    }


def print_summary(all_results: Dict[str, Any]) -> None:
    """Print summary table."""
    print("\n" + "=" * 90)
    print("ADDITIONAL BASELINES RESULTS")
    print("=" * 90)
    
    methods = ["ddm", "eddm", "kswin", "page_hinkley"]
    method_names = ["DDM", "EDDM", "KSWIN", "Page-Hinkley"]
    
    print(f"\n{'Environment':<18}", end="")
    for name in method_names:
        print(f"{name:<18}", end="")
    print()
    print("-" * 90)
    
    for env_name, data in sorted(all_results.items()):
        print(f"{env_name:<18}", end="")
        for method in methods:
            s = data["summary"][method]
            det = f"{s['detection_rate']*100:.0f}%"
            fp = f"FP={s['mean_fp']:.1f}"
            print(f"{det} ({fp})     ", end="")
        print()
    
    # Averages
    print("-" * 90)
    print(f"{'AVERAGE':<18}", end="")
    for method in methods:
        rates = [data["summary"][method]["detection_rate"] for data in all_results.values()]
        fps = [data["summary"][method]["mean_fp"] for data in all_results.values()]
        print(f"{np.mean(rates)*100:.0f}% (FP={np.mean(fps):.1f})     ", end="")
    print()


def merge_results(output_dir: Path) -> Dict[str, Any]:
    """Merge all additional baseline results."""
    files = list(output_dir.glob("additional_baselines_*.json"))
    
    if not files:
        print(f"No additional baseline files found in {output_dir}")
        return {}
    
    print(f"Found {len(files)} result files")
    
    all_results = {}
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        env_name = data["env_name"]
        all_results[env_name] = data
    
    print_summary(all_results)
    
    # Save merged
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_file = output_dir / f"additional_baselines_merged_{timestamp}.json"
    with open(merged_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMerged results saved to: {merged_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run additional baseline experiments")
    parser.add_argument("--env", type=str, help="Environment to test")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--merge", action="store_true", help="Merge all results")
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if args.merge:
        merge_results(output_dir)
        return
    
    if not args.env:
        print("Please specify --env or use --merge")
        print("Available: tictactoe, connectfour, kuhnpoker, maze, tictactoe-pz, connectfour-pz, cartpole-gym")
        return
    
    print("=" * 60)
    print("Additional Baselines Experiment")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Seeds: {args.seeds}")
    print(f"Methods: DDM, EDDM, KSWIN, Page-Hinkley")
    print()
    
    result = run_multi_seed(args.env, n_seeds=args.seeds)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"additional_baselines_{args.env}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print(f"\nSummary for {args.env}:")
    print("-" * 50)
    for method, name in [("ddm", "DDM"), ("eddm", "EDDM"), ("kswin", "KSWIN"), ("page_hinkley", "Page-Hinkley")]:
        s = result["summary"][method]
        det = f"{s['detection_rate']*100:.0f}%"
        delay = f"{s['mean_delay']:.1f}" if s['mean_delay'] else "N/A"
        fp = f"{s['mean_fp']:.1f}"
        print(f"  {name:<12} Detection={det:<5} Delay={delay:<6} FP={fp}")


if __name__ == "__main__":
    main()
