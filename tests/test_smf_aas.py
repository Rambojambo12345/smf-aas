"""
Test suite for SMF-AAS framework.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smf_aas import (
    StrategyMonitor,
    MonitorConfig,
    Alert,
    AlertLevel,
    get_env,
    list_envs,
    GameEnvironment,
    GameState,
)
from smf_aas.metrics import (
    StateDistributionShift,
    BehaviorShiftDetector,
    PerformanceMonitor,
)
from smf_aas.agents import TabularQLearning, RandomAgent
from smf_aas.baselines import CUSUM, PageHinkley, ADWIN, PerformanceOnly


# =============================================================================
# Configuration Tests
# =============================================================================

class TestMonitorConfig:
    """Test MonitorConfig validation."""
    
    def test_valid_config(self):
        config = MonitorConfig(window_size=50, baseline_episodes=200)
        assert config.window_size == 50
        assert config.baseline_episodes == 200
    
    def test_invalid_window_size(self):
        with pytest.raises(ValueError):
            MonitorConfig(window_size=5)  # Too small
    
    def test_invalid_baseline(self):
        with pytest.raises(ValueError):
            MonitorConfig(window_size=50, baseline_episodes=50)  # Too small
    
    def test_invalid_thresholds(self):
        with pytest.raises(ValueError):
            MonitorConfig(yellow_threshold=3.0, red_threshold=2.0)


# =============================================================================
# Metric Component Tests
# =============================================================================

class TestStateDistributionShift:
    """Test S component (Jensen-Shannon divergence)."""
    
    def test_initialization(self):
        detector = StateDistributionShift(window_size=20)
        assert detector.window_size == 20
        assert detector.episode_count == 0
    
    def test_returns_none_until_two_windows(self):
        detector = StateDistributionShift(window_size=10)
        
        # First window
        for _ in range(10):
            result = detector.add_episode([np.zeros(5)])
        
        # Should still be None (need 2 windows)
        assert result is None
    
    def test_detects_distribution_shift(self):
        detector = StateDistributionShift(window_size=10)
        
        # First window: states around 0
        for _ in range(10):
            states = [np.random.randn(5) for _ in range(5)]
            detector.add_episode(states)
        
        # Second window: same distribution
        for _ in range(10):
            states = [np.random.randn(5) for _ in range(5)]
            result = detector.add_episode(states)
        
        assert result is not None
        assert result >= 0  # JS divergence is non-negative
        
        # Third window: shifted distribution
        for _ in range(10):
            states = [np.random.randn(5) + 5.0 for _ in range(5)]
            result = detector.add_episode(states)
        
        assert result is not None
        assert result > 0.1  # Should detect shift
    
    def test_reset(self):
        detector = StateDistributionShift(window_size=10)
        for _ in range(15):
            detector.add_episode([np.zeros(5)])
        
        detector.reset()
        assert detector.episode_count == 0
        assert len(detector.history) == 0


class TestBehaviorShiftDetector:
    """Test B component (behavioral features)."""
    
    def test_initialization(self):
        detector = BehaviorShiftDetector(window_size=20, n_actions=4)
        assert detector.window_size == 20
        assert detector.n_actions == 4
    
    def test_feature_extraction(self):
        detector = BehaviorShiftDetector(window_size=10, n_actions=4)
        
        # Add episodes
        for _ in range(20):
            states = [np.zeros(5) for _ in range(10)]
            actions = [np.random.randint(0, 4) for _ in range(9)]
            rewards = [0.0] * 8 + [1.0]
            detector.add_episode(states, actions, rewards)
        
        assert len(detector.feature_history) >= 1
        features = detector.feature_history[0]
        
        assert 0 <= features.action_entropy <= 1
        assert features.episode_length >= 0
        assert 0 <= features.state_revisitation <= 1
        assert 0 <= features.action_persistence <= 1


class TestPerformanceMonitor:
    """Test P component (Cohen's d)."""
    
    def test_initialization(self):
        monitor = PerformanceMonitor(window_size=20)
        assert monitor.window_size == 20
    
    def test_cohens_d_computation(self):
        monitor = PerformanceMonitor(window_size=10)
        
        # First window
        for _ in range(10):
            monitor.add_episode(1.0)
        
        # Second window (same)
        for _ in range(10):
            result = monitor.add_episode(1.0)
        
        assert result is not None
        assert abs(result) < 0.1  # No difference
        
        # Third window (different)
        for _ in range(10):
            result = monitor.add_episode(5.0)
        
        assert result is not None
        assert abs(result) > 1.0  # Large effect size


# =============================================================================
# Environment Tests
# =============================================================================

class TestEnvironments:
    """Test all internal environments."""
    
    @pytest.mark.parametrize("env_name", ["tictactoe", "connectfour", "kuhnpoker", "maze"])
    def test_environment_exists(self, env_name):
        assert env_name in list_envs()
    
    @pytest.mark.parametrize("env_name", ["tictactoe", "connectfour", "kuhnpoker", "maze"])
    def test_environment_interface(self, env_name):
        env = get_env(env_name)
        
        # Check properties
        assert isinstance(env.name, str)
        assert env.num_actions > 0
        assert env.num_players >= 1
        assert len(env.state_shape) >= 1
        
        # Check reset
        state = env.reset()
        assert isinstance(state, GameState)
        assert isinstance(state.observation, np.ndarray)
        assert len(state.legal_actions) > 0
        assert not state.is_terminal
        
        # Check step
        action = state.legal_actions[0]
        new_state, done = env.step(action)
        assert isinstance(new_state, GameState)
        assert isinstance(done, bool)
    
    @pytest.mark.parametrize("env_name", ["tictactoe", "connectfour", "kuhnpoker", "maze"])
    def test_full_episode(self, env_name):
        env = get_env(env_name)
        state = env.reset()
        
        rng = np.random.RandomState(42)
        steps = 0
        max_steps = 1000
        
        while not state.is_terminal and steps < max_steps:
            action = rng.choice(state.legal_actions)
            state, done = env.step(action)
            steps += 1
        
        assert state.is_terminal
        assert state.returns is not None
    
    @pytest.mark.parametrize("env_name", ["tictactoe", "connectfour", "kuhnpoker", "maze"])
    def test_opponents(self, env_name):
        env = get_env(env_name)
        opponents = env.get_opponents(seed=42)
        
        assert isinstance(opponents, dict)
        assert len(opponents) >= 1
        
        # Test opponent interface
        state = env.reset()
        for name, opp in opponents.items():
            action = opp.get_action(state.observation, state.legal_actions)
            assert action in state.legal_actions


class TestTicTacToe:
    """Specific tests for TicTacToe."""
    
    def test_win_detection(self):
        env = get_env("tictactoe")
        
        # Play a winning game for player 0
        state = env.reset()
        moves = [4, 0, 1, 3, 7]  # P0: 4,1,7 (diagonal), P1: 0,3
        
        for i, move in enumerate(moves):
            state, done = env.step(move)
            if i < 4:
                assert not done
        
        assert done
        assert state.returns[0] == 1.0  # Player 0 wins


class TestConnectFour:
    """Specific tests for ConnectFour."""
    
    def test_column_filling(self):
        env = get_env("connectfour")
        state = env.reset()
        
        # Fill column 0
        for _ in range(6):
            if 0 in state.legal_actions:
                state, done = env.step(0)
        
        assert 0 not in state.legal_actions


# =============================================================================
# Agent Tests
# =============================================================================

class TestAgents:
    """Test learning agents."""
    
    def test_tabular_q_learning(self):
        agent = TabularQLearning(n_actions=4, epsilon=0.1, seed=42)
        
        obs = np.array([0.5, 0.5])
        legal = [0, 1, 2, 3]
        
        action = agent.get_action(obs, legal)
        assert action in legal
        
        agent.update(1.0, obs, legal, False)
        assert agent.get_q_table_size() > 0
    
    def test_random_agent(self):
        agent = RandomAgent(seed=42)
        
        obs = np.array([0.5, 0.5])
        legal = [0, 1, 2]
        
        action = agent.get_action(obs, legal)
        assert action in legal


# =============================================================================
# Baseline Tests
# =============================================================================

class TestBaselines:
    """Test baseline detection methods."""
    
    def test_cusum(self):
        cusum = CUSUM(threshold=5.0, warmup=20)
        
        # Warmup
        for _ in range(20):
            cusum.update(0.0)
        
        # Normal values
        for _ in range(30):
            result = cusum.update(np.random.randn() * 0.1)
        
        # Shifted values
        detected = False
        for _ in range(50):
            result = cusum.update(5.0)
            if result and result.detected:
                detected = True
                break
        
        assert detected
    
    def test_adwin(self):
        adwin = ADWIN(delta=0.01)
        
        # Stable period
        for _ in range(100):
            adwin.update(0.0)
        
        # Detect change
        detected = False
        for _ in range(100):
            result = adwin.update(5.0)
            if result and result.detected:
                detected = True
                break
        
        assert detected


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Full integration tests."""
    
    def test_full_monitoring_pipeline(self):
        """Test complete monitoring pipeline on TicTacToe."""
        # Setup
        env = get_env("tictactoe")
        opponents = env.get_opponents(42)
        opp_list = list(opponents.values())
        
        agent = TabularQLearning(n_actions=9, epsilon=0.1, seed=42)
        
        config = MonitorConfig(
            window_size=20,
            baseline_episodes=60,
        )
        monitor = StrategyMonitor(config, n_actions=9)
        
        # Run episodes
        for ep in range(100):
            opp = opp_list[0] if ep < 70 else opp_list[-1]
            
            states, actions, rewards = [], [], []
            state = env.reset()
            states.append(state.observation.copy())
            agent.reset()
            
            while not state.is_terminal:
                if state.current_player == 0:
                    action = agent.get_action(state.observation, state.legal_actions)
                    actions.append(action)
                    rewards.append(0.0)
                else:
                    action = opp.get_action(state.observation, state.legal_actions)
                
                state, done = env.step(action)
                states.append(state.observation.copy())
            
            ret = state.returns[0] if state.returns else 0.0
            if rewards:
                rewards[-1] = ret
            
            monitor.update(states, actions, rewards, ret)
        
        # Verify baseline was established
        assert monitor.baseline_complete
        assert len(monitor.baseline_stats) == 3  # S, B, P
    
    def test_alert_generation(self):
        """Test that alerts are generated correctly."""
        config = MonitorConfig(
            window_size=10,
            baseline_episodes=30,
            yellow_threshold=1.5,
            red_threshold=2.5,
        )
        monitor = StrategyMonitor(config, n_actions=4)
        
        # Baseline period
        for _ in range(30):
            states = [np.zeros(4)]
            actions = [0]
            rewards = [1.0]
            monitor.update(states, actions, rewards, 1.0)
        
        assert monitor.baseline_complete
        
        # Detection period with change
        for _ in range(20):
            states = [np.ones(4) * 10]  # Very different states
            actions = [3]  # Different action
            rewards = [-5.0]  # Different return
            alert = monitor.update(states, actions, rewards, -5.0)
        
        # Should have generated alerts
        assert len(monitor.alerts) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
