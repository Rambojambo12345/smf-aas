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
    """Check if Gymnasium is available."""
    try:
        import gymnasium as gym
        return True, gym
    except ImportError:
        return False, None


class GymnasiumCartPole(GameEnvironment):
    """Gymnasium CartPole wrapper with configurable wind for dynamics change experiments.
    
    CartPole-v1 is a classic control task where the agent must balance
    a pole on a cart by applying left/right forces.
    
    For monitoring experiments, strategy changes are induced by adding
    wind (a constant force bias) that pushes the cart in one direction.
    
    Parameters
    ----------
    seed : int, optional
        Random seed.
    max_steps : int, default=500
        Maximum steps per episode.
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        max_steps: int = 500
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
        self._wind_force: float = 0.0
    
    @property
    def wind_force(self) -> float:
        """Current wind force."""
        return self._wind_force
    
    def set_wind(self, force: float) -> None:
        """Set wind force for dynamics change experiments.
        
        Parameters
        ----------
        force : float
            Wind force to apply. Positive pushes cart right.
            Typical values: 0.02 to 0.05 for noticeable but manageable wind.
        """
        self._wind_force = force
    
    @property
    def name(self) -> str:
        return "CartPole-Gym"
    
    @property
    def num_actions(self) -> int:
        return 2
    
    @property
    def num_players(self) -> int:
        return 1
    
    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (4,)
    
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
        
        # Apply wind: modify cart velocity (index 1 in observation)
        if self._wind_force != 0.0:
            self._obs = np.array(self._obs)
            self._obs[1] += self._wind_force
        
        done = terminated or truncated or self._steps >= self._max_steps
        
        return GameState(
            observation=np.asarray(self._obs, dtype=np.float32),
            legal_actions=[] if done else [0, 1],
            current_player=0,
            is_terminal=done,
            returns=[float(self._steps)] if done else None
        ), done
    
    def get_opponents(self, seed: int = 42) -> Dict[str, Any]:
        """Get wind configurations for single-player environment."""
        return {
            'no_wind': _WindConfig(self, wind_force=0.0),
            'wind_right': _WindConfig(self, wind_force=0.03),
        }


class _WindConfig:
    """Wrapper that sets wind force when activated."""
    
    def __init__(self, env: GymnasiumCartPole, wind_force: float) -> None:
        self._env = env
        self._wind_force = wind_force
    
    def configure(self) -> None:
        """Configure the environment with this wind setting."""
        self._env.set_wind(self._wind_force)
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        """Return dummy action (not used for single-player)."""
        return legal_actions[0] if legal_actions else 0