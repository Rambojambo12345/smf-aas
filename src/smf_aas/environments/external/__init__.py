"""
External environment wrappers for validation.

These wrappers integrate established packages (PettingZoo, OpenSpiel, Gymnasium)
to demonstrate framework generalization beyond internal implementations.

Requirements:
    pip install pettingzoo gymnasium

"""

import warnings

from ..registry import register_env

# Track available external environments
_EXTERNAL_AVAILABLE = []

# Try PettingZoo TicTacToe
try:
    from .pettingzoo_envs import PettingZooTicTacToe, PettingZooConnectFour
    register_env('tictactoe-pz', PettingZooTicTacToe)
    register_env('connectfour-pz', PettingZooConnectFour)
    _EXTERNAL_AVAILABLE.extend(['tictactoe-pz', 'connectfour-pz'])
except ImportError:
    pass

# Try Gymnasium environments
try:
    from .gymnasium_envs import GymnasiumCartPole
    register_env('cartpole-gym', GymnasiumCartPole)
    _EXTERNAL_AVAILABLE.append('cartpole-gym')
except ImportError:
    pass


def get_available_external() -> list:
    """Get list of available external environments."""
    return _EXTERNAL_AVAILABLE.copy()


__all__ = ['get_available_external']
