"""
Environments module for SMF-AAS experiments.

This module provides:
1. Abstract base classes for game environments
2. Internal reference implementations (for primary validation)
3. External package wrappers (for additional validation)

Environment Strategy
--------------------
The framework uses a two-tier validation approach:

**Internal Environments (Primary)**
    Self-contained implementations for controlled, reproducible experiments.
    These form the basis of the main experimental results.

**External Environments (Validation)**
    Wrappers around established packages (PettingZoo, OpenSpiel) to
    demonstrate generalization beyond internal implementations.

Available Environments
----------------------
Internal (always available):
    - ``tictactoe``: 2-player Tic-Tac-Toe
    - ``connectfour``: 2-player Connect Four  
    - ``kuhnpoker``: 2-player Kuhn Poker (imperfect information)
    - ``maze``: Single-player grid navigation

External (requires additional packages):
    - ``tictactoe-pz``: PettingZoo Tic-Tac-Toe (requires pettingzoo)
    - ``connectfour-pz``: PettingZoo Connect Four (requires pettingzoo)
    - ``kuhnpoker-osp``: OpenSpiel Kuhn Poker (requires open_spiel)

Examples
--------
>>> from smf_aas.environments import get_env, list_envs
>>> print(list_envs())
['tictactoe', 'connectfour', 'kuhnpoker', 'maze']
>>> 
>>> env = get_env('tictactoe')
>>> state = env.reset()
>>> print(state.legal_actions)
[0, 1, 2, 3, 4, 5, 6, 7, 8]
"""

from .base import GameEnvironment, GameState
from .registry import get_env, list_envs, register_env, ENV_REGISTRY

# Import internal implementations to register them
from . import internal

# Try to import external wrappers (optional)
try:
    from . import external
except ImportError:
    pass  # External packages not available

__all__ = [
    # Base classes
    "GameEnvironment",
    "GameState",
    # Factory functions
    "get_env",
    "list_envs",
    "register_env",
    "ENV_REGISTRY",
]
