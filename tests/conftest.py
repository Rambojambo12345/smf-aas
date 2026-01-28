"""
Pytest configuration and fixtures.

This module provides shared fixtures for all tests.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def rng():
    """Provide seeded random number generator."""
    return np.random.RandomState(42)


@pytest.fixture
def sample_states(rng):
    """Provide sample state sequences."""
    return [rng.randn(9).astype(np.float32) for _ in range(10)]


@pytest.fixture
def sample_actions(rng):
    """Provide sample action sequences."""
    return [rng.randint(0, 4) for _ in range(9)]


@pytest.fixture
def sample_rewards():
    """Provide sample reward sequences."""
    return [0.0] * 8 + [1.0]


@pytest.fixture
def monitor_config():
    """Provide default monitor configuration."""
    from smf_aas import MonitorConfig
    return MonitorConfig(
        window_size=20,
        baseline_episodes=60,
        yellow_threshold=2.0,
        red_threshold=3.0,
    )


@pytest.fixture
def tictactoe_env():
    """Provide TicTacToe environment."""
    from smf_aas import get_env
    return get_env('tictactoe')


@pytest.fixture
def trained_agent(tictactoe_env):
    """Provide trained Q-learning agent."""
    from smf_aas.agents import TabularQLearning
    
    agent = TabularQLearning(
        n_actions=tictactoe_env.num_actions,
        epsilon=0.1,
        seed=42
    )
    
    opponents = tictactoe_env.get_opponents(42)
    opp = list(opponents.values())[0]
    
    # Quick training
    for _ in range(100):
        state = tictactoe_env.reset()
        agent.reset()
        
        while not state.is_terminal:
            if state.current_player == 0:
                action = agent.get_action(state.observation, state.legal_actions)
            else:
                action = opp.get_action(state.observation, state.legal_actions)
            
            next_state, done = tictactoe_env.step(action)
            
            if state.current_player == 0:
                reward = next_state.returns[0] if done and next_state.returns else 0.0
                agent.update(
                    reward,
                    next_state.observation,
                    next_state.legal_actions if not done else [],
                    done
                )
            
            state = next_state
    
    return agent
