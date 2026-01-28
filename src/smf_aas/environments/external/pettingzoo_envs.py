"""
PettingZoo environment wrappers.

Wraps PettingZoo's AEC (Agent Environment Cycle) API to the SMF-AAS
GameEnvironment interface for external validation experiments.

Requirements:
    pip install pettingzoo

References:
    https://pettingzoo.farama.org/
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from ..base import GameEnvironment, GameState


def _check_pettingzoo():
    """Check if PettingZoo is available (lazy import)."""
    try:
        from pettingzoo.classic import tictactoe_v3, connect_four_v3
        return True, tictactoe_v3, connect_four_v3
    except ImportError:
        return False, None, None


class PettingZooTicTacToe(GameEnvironment):
    """PettingZoo TicTacToe wrapper for external validation.
    
    This wrapper adapts PettingZoo's TicTacToe AEC environment to
    the SMF-AAS GameEnvironment interface.
    
    Note: This is for VALIDATION ONLY. Primary results use internal
    TicTacToeEnv for reproducibility.
    
    Parameters
    ----------
    seed : int, optional
        Random seed.
    
    Raises
    ------
    ImportError
        If pettingzoo is not installed.
    """
    
    def __init__(self, seed: Optional[int] = None) -> None:
        available, tictactoe_v3, _ = _check_pettingzoo()
        if not available:
            raise ImportError(
                "PettingZoo is not installed. Install with: pip install pettingzoo"
            )
        
        self._seed = seed
        self._env = tictactoe_v3.env()
        self._state_shape: Optional[Tuple[int, ...]] = None
        self._initialize_env()
    
    def _initialize_env(self) -> None:
        """Initialize environment and determine observation shape."""
        self._env.reset(seed=self._seed)
        
        # Get observation shape from first agent
        agent = self._env.agents[0]
        obs = self._env.observe(agent)
        arr = self._obs_to_array(obs)
        self._state_shape = arr.shape
    
    @property
    def name(self) -> str:
        return "TicTacToe-PZ"
    
    @property
    def num_actions(self) -> int:
        return 9
    
    @property
    def num_players(self) -> int:
        return 2
    
    @property
    def state_shape(self) -> Tuple[int, ...]:
        return self._state_shape or (9,)
    
    def reset(self) -> GameState:
        self._env.reset(seed=self._seed)
        return self._make_state()
    
    def step(self, action: int) -> Tuple[GameState, bool]:
        self._env.step(action)
        
        # Check if game ended
        if all(self._env.terminations.get(a, False) for a in self._env.agents):
            return self._make_terminal_state(), True
        
        return self._make_state(), False
    
    def _make_state(self) -> GameState:
        """Create GameState from current PettingZoo state."""
        agent = self._env.agent_selection
        agent_idx = self._env.agents.index(agent)
        obs = self._env.observe(agent)
        
        return GameState(
            observation=self._obs_to_array(obs),
            legal_actions=self._get_legal_actions(obs),
            current_player=agent_idx,
            is_terminal=False,
            returns=None
        )
    
    def _make_terminal_state(self) -> GameState:
        """Create terminal GameState with returns."""
        rewards = [self._env.rewards.get(a, 0.0) for a in self._env.agents]
        
        # Get final observation
        agent = self._env.agents[0]
        obs = self._env.observe(agent)
        
        return GameState(
            observation=self._obs_to_array(obs),
            legal_actions=[],
            current_player=0,
            is_terminal=True,
            returns=rewards
        )
    
    def _obs_to_array(self, obs: Any) -> np.ndarray:
        """Convert PettingZoo observation to flat numpy array."""
        if isinstance(obs, dict):
            if 'observation' in obs:
                arr = np.asarray(obs['observation'], dtype=np.float32)
            else:
                # Flatten all numeric values
                values = []
                for v in obs.values():
                    if isinstance(v, np.ndarray):
                        values.extend(v.flatten().tolist())
                arr = np.array(values, dtype=np.float32)
        else:
            arr = np.asarray(obs, dtype=np.float32)
        
        return arr.flatten()
    
    def _get_legal_actions(self, obs: Any) -> List[int]:
        """Extract legal actions from observation."""
        if isinstance(obs, dict) and 'action_mask' in obs:
            mask = obs['action_mask']
            return [i for i, v in enumerate(mask) if v]
        return list(range(9))
    
    def get_opponents(self, seed: int = 42) -> Dict[str, Any]:
        return {
            'random': _RandomPolicy(seed),
            'center_first': _CenterFirstPolicy(seed + 1),
        }


class PettingZooConnectFour(GameEnvironment):
    """PettingZoo Connect Four wrapper for external validation.
    
    Parameters
    ----------
    seed : int, optional
        Random seed.
    """
    
    def __init__(self, seed: Optional[int] = None) -> None:
        available, _, connect_four_v3 = _check_pettingzoo()
        if not available:
            raise ImportError(
                "PettingZoo is not installed. Install with: pip install pettingzoo"
            )
        
        self._seed = seed
        self._env = connect_four_v3.env()
        self._state_shape: Optional[Tuple[int, ...]] = None
        self._initialize_env()
    
    def _initialize_env(self) -> None:
        self._env.reset(seed=self._seed)
        agent = self._env.agents[0]
        obs = self._env.observe(agent)
        arr = self._obs_to_array(obs)
        self._state_shape = arr.shape
    
    @property
    def name(self) -> str:
        return "ConnectFour-PZ"
    
    @property
    def num_actions(self) -> int:
        return 7
    
    @property
    def num_players(self) -> int:
        return 2
    
    @property
    def state_shape(self) -> Tuple[int, ...]:
        return self._state_shape or (42,)
    
    def reset(self) -> GameState:
        self._env.reset(seed=self._seed)
        return self._make_state()
    
    def step(self, action: int) -> Tuple[GameState, bool]:
        self._env.step(action)
        
        if all(self._env.terminations.get(a, False) for a in self._env.agents):
            return self._make_terminal_state(), True
        
        return self._make_state(), False
    
    def _make_state(self) -> GameState:
        agent = self._env.agent_selection
        agent_idx = self._env.agents.index(agent)
        obs = self._env.observe(agent)
        
        return GameState(
            observation=self._obs_to_array(obs),
            legal_actions=self._get_legal_actions(obs),
            current_player=agent_idx,
            is_terminal=False,
            returns=None
        )
    
    def _make_terminal_state(self) -> GameState:
        rewards = [self._env.rewards.get(a, 0.0) for a in self._env.agents]
        agent = self._env.agents[0]
        obs = self._env.observe(agent)
        
        return GameState(
            observation=self._obs_to_array(obs),
            legal_actions=[],
            current_player=0,
            is_terminal=True,
            returns=rewards
        )
    
    def _obs_to_array(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict) and 'observation' in obs:
            arr = np.asarray(obs['observation'], dtype=np.float32)
        else:
            arr = np.asarray(obs, dtype=np.float32)
        return arr.flatten()
    
    def _get_legal_actions(self, obs: Any) -> List[int]:
        if isinstance(obs, dict) and 'action_mask' in obs:
            mask = obs['action_mask']
            return [i for i, v in enumerate(mask) if v]
        return list(range(7))
    
    def get_opponents(self, seed: int = 42) -> Dict[str, Any]:
        return {
            'random': _RandomPolicy(seed),
            'center_bias': _CenterBiasPolicy(seed + 1),
        }


class _RandomPolicy:
    """Random policy for opponents."""
    
    def __init__(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        return int(self._rng.choice(legal_actions))


class _CenterFirstPolicy:
    """Policy that prefers center positions (for TicTacToe).
    
    Prioritizes: center (4) > corners (0,2,6,8) > edges (1,3,5,7)
    """
    
    def __init__(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        # Priority order for TicTacToe
        priority = [4, 0, 2, 6, 8, 1, 3, 5, 7]
        
        for pos in priority:
            if pos in legal_actions:
                return pos
        
        # Fallback to random
        return int(self._rng.choice(legal_actions))


class _CenterBiasPolicy:
    """Policy that prefers center columns (for ConnectFour).
    
    Prioritizes center columns: 3 > 2,4 > 1,5 > 0,6
    """
    
    def __init__(self, seed: int, center_prob: float = 0.7) -> None:
        self._rng = np.random.RandomState(seed)
        self._center_prob = center_prob
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        if self._rng.random() < self._center_prob:
            # Prefer center columns
            priority = [3, 2, 4, 1, 5, 0, 6]
            for col in priority:
                if col in legal_actions:
                    return col
        
        # Random fallback
        return int(self._rng.choice(legal_actions))
