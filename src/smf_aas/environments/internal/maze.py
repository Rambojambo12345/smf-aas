"""
Maze Navigation Environment.

A simple grid navigation task where an agent must reach a goal position.
This is a single-player environment used to test the framework on
non-adversarial domains.

For monitoring experiments, strategy changes are induced by moving the
goal to different positions, which requires the agent to adapt its
learned policy.

Game Properties
---------------
- Players: 1
- Information: Perfect
- Determinism: Deterministic
- State space: Grid positions
- Action space: 4 (up, down, left, right)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from ..base import GameEnvironment, GameState


class MazeEnv(GameEnvironment):
    """Simple grid maze navigation environment with configurable goal.
    
    The agent starts at position (0, 0) and must reach a configurable
    goal position. The observation includes the current position
    and goal position (normalized to [0, 1]).
    
    For monitoring experiments, different "opponents" are actually
    different goal configurations that test policy adaptation.
    
    Parameters
    ----------
    size : int, default=5
        Grid size (size × size).
    max_steps : int, default=50
        Maximum steps before timeout.
    goal_position : str, default='bottom_right'
        Initial goal position: 'bottom_right', 'top_right', 'bottom_left'.
    seed : int, optional
        Random seed.
    
    Attributes
    ----------
    pos : List[int]
        Current [x, y] position.
    goal : List[int]
        Goal [x, y] position.
    steps : int
        Steps taken in current episode.
    
    Examples
    --------
    >>> env = MazeEnv(size=5)
    >>> state = env.reset()
    >>> state.observation
    array([0. , 0. , 0.8, 0.8], dtype=float32)
    >>> state, done = env.step(1)  # Move down (positive y)
    """
    
    # Actions: 0=up, 1=down, 2=left, 3=right
    _ACTIONS = {
        0: (0, -1),   # Up (negative y)
        1: (0, 1),    # Down (positive y)
        2: (-1, 0),   # Left (negative x)
        3: (1, 0),    # Right (positive x)
    }
    
    # Goal presets (as fractions of grid size)
    _GOAL_PRESETS = {
        'bottom_right': (1.0, 1.0),   # Default: opposite corner
        'top_right': (1.0, 0.0),      # Top-right corner
        'bottom_left': (0.0, 1.0),    # Bottom-left corner
        'center': (0.5, 0.5),         # Center of grid
    }
    
    def __init__(
        self,
        size: int = 5,
        max_steps: int = 50,
        goal_position: str = 'bottom_right',
        seed: Optional[int] = None
    ) -> None:
        if size < 3:
            raise ValueError("size must be at least 3")
        
        self._size = size
        self._max_steps = max_steps
        self._goal_position = goal_position
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        
        self._pos: List[int] = [0, 0]
        self._goal: List[int] = self._compute_goal(goal_position)
        self._steps: int = 0
        self._done: bool = False
    
    def _compute_goal(self, position: str) -> List[int]:
        """Compute goal coordinates from preset name."""
        if position not in self._GOAL_PRESETS:
            raise ValueError(f"Unknown goal position: {position}")
        
        fx, fy = self._GOAL_PRESETS[position]
        gx = int(fx * (self._size - 1))
        gy = int(fy * (self._size - 1))
        
        # Ensure goal is not at start
        if gx == 0 and gy == 0:
            gx = self._size - 1
            gy = self._size - 1
        
        return [gx, gy]
    
    def set_goal_position(self, position: str) -> None:
        """Change the goal position.
        
        This is used to induce strategy changes in monitoring experiments.
        
        Parameters
        ----------
        position : str
            New goal position preset.
        """
        self._goal_position = position
        self._goal = self._compute_goal(position)
    
    @property
    def name(self) -> str:
        return f"Maze-{self._size}x{self._size}"
    
    @property
    def num_actions(self) -> int:
        return 4
    
    @property
    def num_players(self) -> int:
        return 1
    
    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (4,)  # [pos_x, pos_y, goal_x, goal_y] normalized
    
    def reset(self) -> GameState:
        """Reset to start position."""
        self._pos = [0, 0]
        self._steps = 0
        self._done = False
        return self._make_state()
    
    def step(self, action: int) -> Tuple[GameState, bool]:
        """Execute movement action."""
        if self._done:
            raise ValueError("Episode is already finished")
        
        if action not in self._ACTIONS:
            raise ValueError(f"Invalid action {action}: must be 0-3")
        
        self._steps += 1
        
        # Apply movement
        dx, dy = self._ACTIONS[action]
        new_x = max(0, min(self._size - 1, self._pos[0] + dx))
        new_y = max(0, min(self._size - 1, self._pos[1] + dy))
        self._pos = [new_x, new_y]
        
        # Check terminal conditions
        reached_goal = self._pos == self._goal
        timeout = self._steps >= self._max_steps
        
        if reached_goal:
            self._done = True
            return self._make_state(is_terminal=True, returns=[1.0]), True
        
        if timeout:
            self._done = True
            return self._make_state(is_terminal=True, returns=[-1.0]), True
        
        return self._make_state(), False
    
    def _make_state(
        self,
        is_terminal: bool = False,
        returns: Optional[List[float]] = None
    ) -> GameState:
        """Create normalized observation."""
        obs = np.array([
            self._pos[0] / (self._size - 1) if self._size > 1 else 0,
            self._pos[1] / (self._size - 1) if self._size > 1 else 0,
            self._goal[0] / (self._size - 1) if self._size > 1 else 0,
            self._goal[1] / (self._size - 1) if self._size > 1 else 0,
        ], dtype=np.float32)
        
        return GameState(
            observation=obs,
            legal_actions=[] if is_terminal else [0, 1, 2, 3],
            current_player=0,
            is_terminal=is_terminal,
            returns=returns
        )
    
    def get_opponents(self, seed: int = 42) -> Dict[str, Any]:
        """Get goal configurations for single-player environment.
        
        For maze, "opponents" are different goal positions that require
        the agent to adapt its policy. This creates meaningful strategy
        changes in single-player settings.
        
        Returns
        -------
        Dict[str, Any]
            Mapping of configuration names to _GoalConfig objects.
        """
        return {
            'goal_bottom_right': _GoalConfig(self, 'bottom_right'),
            'goal_top_right': _GoalConfig(self, 'top_right'),
        }


class _GoalConfig:
    """Wrapper that sets goal position when get_action is called.
    
    This allows the maze to be used with the same experiment interface
    as adversarial games, where "opponent" changes trigger env changes.
    """
    
    def __init__(self, env: MazeEnv, goal_position: str) -> None:
        self._env = env
        self._goal_position = goal_position
        self._configured = False
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        """Configure goal and return agent's best action.
        
        Note: For single-player, this just ensures goal is set correctly.
        The actual action is taken by the learning agent, not this "opponent".
        """
        if not self._configured:
            self._env.set_goal_position(self._goal_position)
            self._configured = True
        # Return any legal action (won't be used for single-player)
        return legal_actions[0] if legal_actions else 0
    
    def reset(self) -> None:
        """Reset configuration state for new episode."""
        self._configured = False
