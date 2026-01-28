"""
Tic-Tac-Toe Environment.

A classic 3x3 board game for two players. Players alternate placing marks
(X and O) on empty cells, with the goal of getting three in a row.

Game Properties
---------------
- Players: 2
- Information: Perfect
- Determinism: Deterministic
- State space: 3^9 = 19,683 (theoretical maximum)
- Action space: 9 (cells 0-8)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from ..base import GameEnvironment, GameState


# Winning line indices
_WINNING_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
    [0, 4, 8], [2, 4, 6]              # Diagonals
]


class TicTacToeEnv(GameEnvironment):
    """Tic-Tac-Toe environment.
    
    The board is represented as a 9-element array:
        0 | 1 | 2
        ---------
        3 | 4 | 5
        ---------
        6 | 7 | 8
    
    Board values:
        0.0 = empty
        1.0 = player 0 (X)
       -1.0 = player 1 (O)
    
    Parameters
    ----------
    seed : int, optional
        Random seed for opponent policies.
    
    Attributes
    ----------
    board : np.ndarray
        Current board state.
    current : int
        Current player (0 or 1).
    
    Examples
    --------
    >>> env = TicTacToeEnv()
    >>> state = env.reset()
    >>> state.legal_actions
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> state, done = env.step(4)  # Play center
    >>> 4 not in state.legal_actions
    True
    """
    
    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._board: np.ndarray = np.zeros(9, dtype=np.float32)
        self._current: int = 0
        self._done: bool = False
    
    @property
    def name(self) -> str:
        return "TicTacToe"
    
    @property
    def num_actions(self) -> int:
        return 9
    
    @property
    def num_players(self) -> int:
        return 2
    
    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (9,)
    
    def reset(self) -> GameState:
        """Reset to empty board with player 0 to move."""
        self._board = np.zeros(9, dtype=np.float32)
        self._current = 0
        self._done = False
        return self._make_state()
    
    def step(self, action: int) -> Tuple[GameState, bool]:
        """Execute action (place mark on cell)."""
        if self._done:
            raise ValueError("Game is already finished")
        
        if action < 0 or action > 8:
            raise ValueError(f"Invalid action {action}: must be 0-8")
        
        if self._board[action] != 0:
            raise ValueError(f"Invalid action {action}: cell is occupied")
        
        # Place mark
        self._board[action] = 1.0 if self._current == 0 else -1.0
        
        # Check for winner
        winner = self._check_winner()
        
        if winner is not None:
            self._done = True
            # Winner gets +1, loser gets -1
            if winner == 1:
                returns = [1.0, -1.0]
            else:
                returns = [-1.0, 1.0]
            return self._make_state(is_terminal=True, returns=returns), True
        
        # Check for draw
        if len(self._get_legal_actions()) == 0:
            self._done = True
            return self._make_state(is_terminal=True, returns=[0.0, 0.0]), True
        
        # Switch player
        self._current = 1 - self._current
        return self._make_state(), False
    
    def _make_state(
        self,
        is_terminal: bool = False,
        returns: Optional[List[float]] = None
    ) -> GameState:
        """Create GameState from current internal state."""
        return GameState(
            observation=self._board.copy(),
            legal_actions=[] if is_terminal else self._get_legal_actions(),
            current_player=self._current,
            is_terminal=is_terminal,
            returns=returns
        )
    
    def _get_legal_actions(self) -> List[int]:
        """Get list of empty cells."""
        return [i for i in range(9) if self._board[i] == 0]
    
    def _check_winner(self) -> Optional[int]:
        """Check if there's a winner. Returns 1 (player 0) or -1 (player 1)."""
        for line in _WINNING_LINES:
            line_sum = sum(self._board[i] for i in line)
            if line_sum == 3:
                return 1   # Player 0 wins
            if line_sum == -3:
                return -1  # Player 1 wins
        return None
    
    def get_opponents(self, seed: int = 42) -> Dict[str, Any]:
        """Get opponent policies for testing.
        
        Returns
        -------
        Dict[str, Any]
            - 'random': Uniform random over legal actions
            - 'center_first': Prefers center, then corners
        """
        return {
            'random': _RandomPolicy(seed),
            'center_first': _CenterFirstPolicy(seed + 1),
        }


class _RandomPolicy:
    """Uniform random policy over legal actions."""
    
    def __init__(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        if not legal_actions:
            raise ValueError("No legal actions available")
        return int(self._rng.choice(legal_actions))


class _CenterFirstPolicy:
    """Policy that prefers center, then corners, then edges."""
    
    _CENTER = 4
    _CORNERS = [0, 2, 6, 8]
    _EDGES = [1, 3, 5, 7]
    
    def __init__(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        # Prefer center
        if self._CENTER in legal_actions:
            return self._CENTER
        
        # Then corners
        corners = [c for c in self._CORNERS if c in legal_actions]
        if corners:
            return int(self._rng.choice(corners))
        
        # Then edges
        edges = [e for e in self._EDGES if e in legal_actions]
        if edges:
            return int(self._rng.choice(edges))
        
        # Fallback
        return int(self._rng.choice(legal_actions))
