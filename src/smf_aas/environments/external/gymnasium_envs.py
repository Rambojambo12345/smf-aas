"""
Gymnasium environment wrappers.

Wraps Gymnasium environments for single-player RL validation.

Requirements:
    pip install gymnasium
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from ..base import GameEnvironment, GameState


def _check_gymnasium():
    """Check if Gymnasium is available (lazy import)."""
    try:
        import gymnasium as gym
        return True, gym
    except ImportError:
        return False, None


class GymnasiumCartPole(GameEnvironment):
    """Gymnasium CartPole wrapper for single-player validation.
    
    CartPole-v1 is a classic control task where the agent must balance
    a pole on a cart by applying left/right forces.
    
    Parameters
    ----------
    seed : int, optional
        Random seed.
    max_steps : int, default=1500
        Maximum steps per episode.
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        max_steps: int = 1500
    ) -> None:
        available, gym = _check_gymnasium()
        if not available:
            raise ImportError(
                "Gymnasium is not installed. Install with: pip install gymnasium"
            )
        
        self._seed = seed
        self._max_steps = max_steps
        self._env = gym.make('CartPole-v1')
        self._steps = 0
        self._obs: Optional[np.ndarray] = None
    
    @property
    def name(self) -> str:
        return "CartPole-Gym"
    
    @property
    def num_actions(self) -> int:
        return 2  # left, right
    
    @property
    def num_players(self) -> int:
        return 1
    
    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (4,)  # cart_pos, cart_vel, pole_angle, pole_vel
    
    def reset(self) -> GameState:
        self._obs, _ = self._env.reset(seed=self._seed)
        self._steps = 0
        
        return GameState(
            observation=np.asarray(self._obs, dtype=np.float32),
            legal_actions=[0, 1],
            current_player=0,
            is_terminal=False,
            returns=None
        )
    
    def step(self, action: int) -> Tuple[GameState, bool]:
        self._obs, reward, terminated, truncated, _ = self._env.step(action)
        self._steps += 1
        
        done = terminated or truncated or self._steps >= self._max_steps
        
        return GameState(
            observation=np.asarray(self._obs, dtype=np.float32),
            legal_actions=[] if done else [0, 1],
            current_player=0,
            is_terminal=done,
            returns=[float(self._steps)] if done else None
        ), done
    
    def get_opponents(self, seed: int = 42) -> Dict[str, Any]:
        """For single-player, return different exploration policies."""
        return {
            'random': _RandomPolicy(seed),
            'biased_left': _BiasedPolicy(seed, bias_action=0),
            'biased_right': _BiasedPolicy(seed, bias_action=1),
        }


class _RandomPolicy:
    def __init__(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        return int(self._rng.choice(legal_actions))


class _BiasedPolicy:
    def __init__(self, seed: int, bias_action: int, bias_prob: float = 0.7) -> None:
        self._rng = np.random.RandomState(seed)
        self._bias_action = bias_action
        self._bias_prob = bias_prob
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        if self._rng.random() < self._bias_prob and self._bias_action in legal_actions:
            return self._bias_action
        return int(self._rng.choice(legal_actions))
