"""
Internal environment implementations.

These are self-contained reference implementations used for primary
experimental validation. They do not depend on external packages.

Environments
------------
- TicTacToe: Classic 3x3 board game
- ConnectFour: 6x7 board game  
- KuhnPoker: Simplified poker with imperfect information
- Maze: Grid navigation task
"""

from .tictactoe import TicTacToeEnv
from .connectfour import ConnectFourEnv
from .kuhnpoker import KuhnPokerEnv
from .maze import MazeEnv

from ..registry import register_env

# Register all internal environments
register_env('tictactoe', TicTacToeEnv)
register_env('connectfour', ConnectFourEnv)
register_env('kuhnpoker', KuhnPokerEnv)
register_env('maze', MazeEnv)

__all__ = [
    'TicTacToeEnv',
    'ConnectFourEnv', 
    'KuhnPokerEnv',
    'MazeEnv',
]
