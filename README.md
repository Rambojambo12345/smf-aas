# SMF-AAS: Strategy Monitoring Framework for Autonomous Agent Systems

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A framework for detecting strategy changes in autonomous agents through multi-component behavioral analysis. SMF-AAS monitors state distributions, behavioral patterns, and performance metrics to identify when an agent's strategy has shifted.

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running Experiments](#running-experiments)
- [Generating Figures](#generating-figures)
- [Framework Overview](#framework-overview)
- [Results](#results)
- [API Reference](#api-reference)
- [Citation](#citation)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Rambojambo12345/smf-aas.git
cd smf-aas
```

### Step 2: Install Core Dependencies

```bash
pip install numpy scipy matplotlib
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Step 3: Install Optional Dependencies (for External Validation)

For PettingZoo environments (TicTacToe-PZ, ConnectFour-PZ):

```bash
pip install "pettingzoo[classic]"
```

For Gymnasium environments (CartPole):

```bash
pip install "gymnasium[classic-control]"
```

Or install everything at once:

```bash
pip install -r requirements.txt
pip install "pettingzoo[classic]" "gymnasium[classic-control]"
```

### Step 4: Verify Installation

```bash
python -c "import sys; sys.path.insert(0, 'src'); from smf_aas import StrategyMonitor, get_env, list_envs; print('Available:', list_envs())"
```

Expected output:
```
Available: ['tictactoe', 'connectfour', 'kuhnpoker', 'maze', 'tictactoe-pz', 'connectfour-pz', 'cartpole-gym']
```

---

## Quick Start

### Basic Usage

```python
import sys
sys.path.insert(0, 'src')

from smf_aas import StrategyMonitor, MonitorConfig, get_env
from smf_aas.agents import TabularQLearning
import numpy as np

# 1. Create environment
env = get_env('tictactoe')

# 2. Create and train an agent
agent = TabularQLearning(n_actions=env.num_actions, epsilon=0.1, seed=42)

# 3. Setup monitor
config = MonitorConfig(window_size=50, baseline_episodes=200)
monitor = StrategyMonitor(config, n_actions=env.num_actions)

# 4. Get opponents (for strategy change simulation)
opponents = env.get_opponents()
opp_list = list(opponents.values())

# 5. Run episodes and monitor
for episode in range(500):
    # Switch opponent at episode 300 (simulates strategy change)
    current_opp = opp_list[0] if episode < 300 else opp_list[1]
    
    state = env.reset()
    states, actions, rewards = [state.observation.copy()], [], []
    
    while not state.is_terminal:
        if state.current_player == 0:
            action = agent.get_action(state.observation, state.legal_actions)
            actions.append(action)
            rewards.append(0.0)
        else:
            action = current_opp.get_action(state.observation, state.legal_actions)
        
        state, _ = env.step(action)
        states.append(state.observation.copy())
    
    # Get episode return
    episode_return = state.returns[0] if state.returns else 0.0
    if rewards:
        rewards[-1] = episode_return
    
    # Update monitor
    alert = monitor.update(states, actions, rewards, episode_return)
    
    if alert:
        print(f"Episode {episode}: {alert.level.value} alert! CDS={alert.composite_score:.2f}")
```

### One-Liner Test

```bash
python -c "import sys; sys.path.insert(0, 'src'); from smf_aas import StrategyMonitor, MonitorConfig; m = StrategyMonitor(MonitorConfig(), n_actions=9); print('SMF-AAS is working!')"
```

---

## Running Experiments

### Run All Internal Environments (No External Packages Needed)

```bash
python experiments/run_main_experiment.py --internal-only --seeds 5
```

### Run Specific Environments

```bash
# Single environment
python experiments/run_main_experiment.py --env tictactoe --seeds 5

# Multiple environments
python experiments/run_main_experiment.py --envs tictactoe connectfour kuhnpoker --seeds 5
```

### Run External Environments (Requires pettingzoo[classic])

```bash
# First install PettingZoo with classic games
pip install "pettingzoo[classic]"

# Then run
python experiments/run_main_experiment.py --envs tictactoe-pz connectfour-pz --seeds 5
```

### Run All Environments (Internal + External)

```bash
# Install all optional dependencies first
pip install "pettingzoo[classic]" "gymnasium[classic-control]"

# Run everything
python experiments/run_main_experiment.py --all --seeds 5
```

### Full Experiment Options

```bash
python experiments/run_main_experiment.py --help
```

```
usage: run_main_experiment.py [-h] [--env ENV] [--all] [--envs ENVS [ENVS ...]]
                              [--seeds SEEDS] [--episodes EPISODES]
                              [--change-point CHANGE_POINT] [--output OUTPUT]
                              [--internal-only] [--external-only]

options:
  --env ENV             Single environment to test
  --envs ENVS [ENVS ...]
                        Multiple environments (e.g., --envs tictactoe connectfour)
  --all                 Run on all available environments
  --internal-only       Run only internal environments (no external packages)
  --external-only       Run only external environments (requires packages)
  --seeds SEEDS         Number of random seeds (default: 10)
  --episodes EPISODES   Total episodes per run (default: 500)
  --change-point CHANGE_POINT
                        Episode where strategy change occurs (default: 300)
  --output OUTPUT       Output directory (default: results)
```

### Example: Complete Experiment Run

```bash
# Step 1: Run internal environments (primary validation)
python experiments/run_main_experiment.py --internal-only --seeds 10

# Step 2: Install external packages
pip install "pettingzoo[classic]"

# Step 3: Run external environments (secondary validation)
python experiments/run_main_experiment.py --external-only --seeds 10

# Results are saved to results/experiment_results_YYYYMMDD_HHMMSS.json
```

---

## Framework Overview

### Architecture

```
smf-aas/
├── src/smf_aas/
│   ├── __init__.py           # Package exports
│   ├── monitor.py            # Core StrategyMonitor class
│   ├── metrics/
│   │   ├── state_shift.py    # S component (Jensen-Shannon divergence)
│   │   ├── behavior_shift.py # B component (behavioral features)
│   │   └── performance.py    # P component (Cohen's d effect size)
│   ├── environments/
│   │   ├── internal/         # Self-contained game environments
│   │   │   ├── tictactoe.py
│   │   │   ├── connectfour.py
│   │   │   ├── kuhnpoker.py
│   │   │   └── maze.py
│   │   └── external/         # Wrappers for PettingZoo/Gymnasium
│   ├── agents.py             # TabularQLearning, RandomAgent
│   └── baselines.py          # CUSUM, ADWIN, PageHinkley
├── experiments/
│   ├── run_main_experiment.py
│   ├── generate_paper_figures.py
│   └── generate_timeseries_figures.py
└── tests/
```

### Components

| Component | Metric | Description |
|-----------|--------|-------------|
| **S** (State Shift) | Jensen-Shannon Divergence | Detects changes in state visitation patterns |
| **B** (Behavior Shift) | Feature Distance | Detects changes in action entropy, episode length, reward patterns |
| **P** (Performance) | Cohen's d Effect Size | Detects changes in episode returns |

### Composite Drift Score (CDS)

```
CDS = √(z_S² + z_B² + z_P²)
```

Where z_S, z_B, z_P are z-scores relative to baseline statistics.

### Alert Levels

| Level | Threshold | Meaning |
|-------|-----------|---------|
| GREEN | CDS < 2.0 | Normal operation |
| YELLOW | 2.0 ≤ CDS < 3.0 | Possible drift detected |
| RED | CDS ≥ 3.0 | Significant drift detected |

---

## API Reference

### StrategyMonitor

```python
from smf_aas import StrategyMonitor, MonitorConfig

# Configuration
config = MonitorConfig(
    window_size=50,           # Episodes per analysis window
    baseline_episodes=200,    # Episodes for baseline statistics
    yellow_threshold=2.0,     # CDS threshold for yellow alert
    red_threshold=3.0,        # CDS threshold for red alert
)

# Initialize monitor
monitor = StrategyMonitor(config, n_actions=9)

# Update with episode data
alert = monitor.update(
    states,          # List of state observations
    actions,         # List of actions taken
    rewards,         # List of rewards received
    episode_return   # Total episode return
)

# Check alert
if alert:
    print(f"Alert: {alert.level.value}, CDS: {alert.composite_score:.2f}")
```

### Environments

```python
from smf_aas import get_env, list_envs

# List available environments
print(list_envs())
# ['tictactoe', 'connectfour', 'kuhnpoker', 'maze', 'tictactoe-pz', ...]

# Get environment
env = get_env('tictactoe')

# Environment properties
print(env.name)           # 'TicTacToe'
print(env.num_actions)    # 9
print(env.num_players)    # 2
print(env.state_shape)    # (9,)

# Get opponents for strategy change simulation
opponents = env.get_opponents()
print(opponents.keys())   # dict_keys(['random', 'center_first'])

# Run episode
state = env.reset()
while not state.is_terminal:
    action = agent.get_action(state.observation, state.legal_actions)
    state, done = env.step(action)
```

### Agents

```python
from smf_aas.agents import TabularQLearning, RandomAgent

# Q-Learning agent
agent = TabularQLearning(
    n_actions=9,
    learning_rate=0.1,
    discount=0.99,
    epsilon=0.1,
    seed=42
)

# Get action
action = agent.get_action(observation, legal_actions)

# Update after transition
agent.update(reward, next_observation, next_legal_actions, done)

# Random agent (for baseline)
random_agent = RandomAgent(n_actions=9, seed=42)
```

### Baselines

```python
from smf_aas.baselines import CUSUM, ADWIN, PerformanceOnly

# CUSUM detector
cusum = CUSUM(threshold=5.0, drift=0.05)
cusum.update(value)
if cusum.detected:
    print(f"Change detected at index {cusum.detection_point}")

# ADWIN detector
adwin = ADWIN(delta=0.002)
adwin.update(value)
if adwin.detected:
    print("Change detected")
```

---

## Citation

If you use SMF-AAS in your research, please cite:

```bibtex
@article{smf_aas_2025,
  title={SMF-AAS: A Multi-Component Framework for Strategy Change Detection in Autonomous Agent Systems},
  author={B.Kleibrink},
  journal={},
  year={2026},
  note={}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
