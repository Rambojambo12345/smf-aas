"""
Environment registry and factory functions.

This module provides a central registry for all available environments
and factory functions to instantiate them by name.
"""

from __future__ import annotations

from typing import Dict, List, Type, Any

from .base import GameEnvironment


# Global environment registry
ENV_REGISTRY: Dict[str, Type[GameEnvironment]] = {}


def register_env(name: str, env_class: Type[GameEnvironment]) -> None:
    """Register an environment class.
    
    Parameters
    ----------
    name : str
        Name to register the environment under (case-insensitive).
    env_class : Type[GameEnvironment]
        Environment class to register.
    
    Raises
    ------
    ValueError
        If name is already registered.
    TypeError
        If env_class is not a subclass of GameEnvironment.
    
    Examples
    --------
    >>> from smf_aas.environments import register_env, GameEnvironment
    >>> class MyEnv(GameEnvironment):
    ...     # implementation
    ...     pass
    >>> register_env('myenv', MyEnv)
    """
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    
    if not isinstance(env_class, type) or not issubclass(env_class, GameEnvironment):
        raise TypeError("env_class must be a subclass of GameEnvironment")
    
    key = name.lower()
    if key in ENV_REGISTRY:
        # Allow re-registration (for testing/development)
        pass
    
    ENV_REGISTRY[key] = env_class


def get_env(name: str, **kwargs: Any) -> GameEnvironment:
    """Create environment instance by name.
    
    Parameters
    ----------
    name : str
        Environment name (case-insensitive).
    **kwargs
        Additional arguments passed to environment constructor.
    
    Returns
    -------
    GameEnvironment
        Instantiated environment.
    
    Raises
    ------
    ValueError
        If environment name is not registered.
    
    Examples
    --------
    >>> env = get_env('tictactoe')
    >>> state = env.reset()
    >>> env.name
    'TicTacToe'
    """
    key = name.lower()
    
    if key not in ENV_REGISTRY:
        available = list(ENV_REGISTRY.keys())
        raise ValueError(
            f"Unknown environment: '{name}'. "
            f"Available: {available if available else '(none registered)'}"
        )
    
    return ENV_REGISTRY[key](**kwargs)


def list_envs() -> List[str]:
    """List all registered environment names.
    
    Returns
    -------
    List[str]
        Sorted list of registered environment names.
    
    Examples
    --------
    >>> list_envs()
    ['connectfour', 'kuhnpoker', 'maze', 'tictactoe']
    """
    return sorted(ENV_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """Check if environment is registered.
    
    Parameters
    ----------
    name : str
        Environment name (case-insensitive).
    
    Returns
    -------
    bool
        True if environment is registered.
    """
    return name.lower() in ENV_REGISTRY
