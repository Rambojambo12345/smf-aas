"""
Connect Four Environment.

A classic board game where two players drop discs into a 6-row, 7-column
grid. The first player to connect four discs in a row (horizontally,
vertically, or diagonally) wins.

Game Properties
---------------
- Players: 2
- Information: Perfect
- Determinism: Deterministic
- Board: 6 rows × 7 columns
- Action space: 7 (columns 0-6)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from ..base import GameEnvironment, GameState


class ConnectFourEnv(GameEnvironment):
    """Connect Four environment.
    
    The board is represented as a 6×7 array (row 0 is top, row 5 is bottom).
    
    Board values:
        0.0 = empty
        1.0 = player 0 (Red)
       -1.0 = player 1 (Yellow)
    
    Actions correspond to column indices (0-6). Discs fall to the lowest
    empty row in the selected column.
    
    Parameters
    ----------
    seed : int, optional
        Random seed for opponent policies.
    
    Examples
    --------
    >>> env = ConnectFourEnv()
    >>> state = env.reset()
    >>> state.legal_actions
    [0, 1, 2, 3, 4, 5, 6]
    >>> state, done = env.step(3)  # Drop in center column
    """
    
    _ROWS = 6
    _COLS = 7
    
    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._board: np.ndarray = np.zeros((self._ROWS, self._COLS), dtype=np.float32)
        self._current: int = 0
        self._done: bool = False
        self._last_move: Optional[Tuple[int, int]] = None
    
    @property
    def name(self) -> str:
        return "ConnectFour"
    
    @property
    def num_actions(self) -> int:
        return self._COLS
    
    @property
    def num_players(self) -> int:
        return 2
    
    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self._ROWS * self._COLS,)  # Flattened
    
    def reset(self) -> GameState:
        """Reset to empty board."""
        self._board = np.zeros((self._ROWS, self._COLS), dtype=np.float32)
        self._current = 0
        self._done = False
        self._last_move = None
        return self._make_state()
    
    def step(self, action: int) -> Tuple[GameState, bool]:
        """Drop disc in column."""
        if self._done:
            raise ValueError("Game is already finished")
        
        if action < 0 or action >= self._COLS:
            raise ValueError(f"Invalid action {action}: must be 0-{self._COLS-1}")
        
        # Find lowest empty row
        row = self._get_drop_row(action)
        if row is None:
            raise ValueError(f"Invalid action {action}: column is full")
        
        # Place disc
        self._board[row, action] = 1.0 if self._current == 0 else -1.0
        self._last_move = (row, action)
        
        # Check for winner (only need to check around last move)
        winner = self._check_winner(row, action)
        
        if winner is not None:
            self._done = True
            returns = [1.0, -1.0] if winner == 1 else [-1.0, 1.0]
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
        return GameState(
            observation=self._board.flatten().copy(),
            legal_actions=[] if is_terminal else self._get_legal_actions(),
            current_player=self._current,
            is_terminal=is_terminal,
            returns=returns
        )
    
    def _get_legal_actions(self) -> List[int]:
        """Get columns that aren't full."""
        return [c for c in range(self._COLS) if self._board[0, c] == 0]
    
    def _get_drop_row(self, col: int) -> Optional[int]:
        """Get the lowest empty row in column, or None if full."""
        for row in range(self._ROWS - 1, -1, -1):
            if self._board[row, col] == 0:
                return row
        return None
    
    def _check_winner(self, row: int, col: int) -> Optional[int]:
        """Check for winner after placing at (row, col)."""
        piece = self._board[row, col]
        if piece == 0:
            return None
        
        # Check all four directions from the last placed piece
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal /
            (1, -1),  # Diagonal \
        ]
        
        for dr, dc in directions:
            count = 1
            # Count in positive direction
            r, c = row + dr, col + dc
            while 0 <= r < self._ROWS and 0 <= c < self._COLS and self._board[r, c] == piece:
                count += 1
                r += dr
                c += dc
            # Count in negative direction
            r, c = row - dr, col - dc
            while 0 <= r < self._ROWS and 0 <= c < self._COLS and self._board[r, c] == piece:
                count += 1
                r -= dr
                c -= dc
            
            if count >= 4:
                return 1 if piece == 1 else -1
        
        return None
    
    def get_opponents(self, seed: int = 42) -> Dict[str, Any]:
        """Get opponent policies."""
        return {
            'random': _RandomPolicy(seed),
            'center_bias': _CenterBiasPolicy(seed + 1),
        }


class _RandomPolicy:
    """Uniform random over legal actions."""
    
    def __init__(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        return int(self._rng.choice(legal_actions))


class _CenterBiasPolicy:
    """Policy that prefers center columns."""
    
    # Columns ordered by preference (center first)
    _PRIORITY = [3, 2, 4, 1, 5, 0, 6]
    
    def __init__(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        for col in self._PRIORITY:
            if col in legal_actions:
                return col
        return int(self._rng.choice(legal_actions))
