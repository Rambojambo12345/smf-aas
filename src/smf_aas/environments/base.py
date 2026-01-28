"""
Base classes for game environments.

This module defines the abstract interface that all environments must implement.
The design follows the principle of minimal, well-defined interfaces that
support both single-player and multi-player environments.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional


@dataclass
class GameState:
    """Immutable game state representation.
    
    This dataclass provides a unified interface for representing game states
    across different environment types (single-player, multi-player, perfect
    information, imperfect information).
    
    Parameters
    ----------
    observation : np.ndarray
        Current observation as a numpy array. For multi-player games with
        imperfect information, this is the observation from the perspective
        of the current player.
    legal_actions : List[int]
        List of legal action indices. Empty list indicates terminal state
        or no available actions.
    current_player : int
        Index of the current player (0 for single-player environments).
    is_terminal : bool
        Whether the game has ended.
    returns : Optional[List[float]]
        Final returns for each player. Only set when is_terminal is True.
        For single-player environments, this is a single-element list.
    
    Examples
    --------
    >>> state = GameState(
    ...     observation=np.zeros(9),
    ...     legal_actions=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    ...     current_player=0,
    ...     is_terminal=False
    ... )
    >>> state.is_terminal
    False
    """
    observation: np.ndarray
    legal_actions: List[int]
    current_player: int
    is_terminal: bool
    returns: Optional[List[float]] = None
    
    def __post_init__(self) -> None:
        """Validate state consistency."""
        if self.is_terminal and not self.legal_actions == []:
            # Terminal states should have no legal actions
            # (but we don't enforce this to allow flexibility)
            pass
        if self.is_terminal and self.returns is None:
            # Terminal states should have returns
            # (but we don't enforce this to allow flexibility)
            pass


class GameEnvironment(ABC):
    """Abstract base class for game environments.
    
    All environments must implement this interface to be compatible with
    the SMF-AAS framework. The interface supports:
    
    - Single-player and multi-player games
    - Perfect and imperfect information
    - Discrete action spaces
    
    Subclasses must implement all abstract methods and properties.
    
    Examples
    --------
    >>> class MyEnv(GameEnvironment):
    ...     @property
    ...     def name(self) -> str:
    ...         return "MyEnv"
    ...     # ... implement other abstract methods
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable environment name.
        
        Returns
        -------
        str
            Environment name (e.g., "TicTacToe", "KuhnPoker").
        """
        pass
    
    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Total number of possible actions.
        
        Returns
        -------
        int
            Size of the action space. For environments with varying action
            spaces per state, this is the maximum number of actions.
        """
        pass
    
    @property
    @abstractmethod
    def num_players(self) -> int:
        """Number of players.
        
        Returns
        -------
        int
            1 for single-player, 2+ for multi-player.
        """
        pass
    
    @property
    @abstractmethod
    def state_shape(self) -> Tuple[int, ...]:
        """Shape of state observation array.
        
        Returns
        -------
        Tuple[int, ...]
            Shape of the observation array (e.g., (9,) for TicTacToe).
        """
        pass
    
    @abstractmethod
    def reset(self) -> GameState:
        """Reset environment to initial state.
        
        Returns
        -------
        GameState
            Initial game state with observation, legal actions, etc.
        """
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[GameState, bool]:
        """Execute action and return new state.
        
        Parameters
        ----------
        action : int
            Action index to execute. Must be in legal_actions of current state.
        
        Returns
        -------
        Tuple[GameState, bool]
            Tuple of (new_state, is_terminal).
        
        Raises
        ------
        ValueError
            If action is not legal in current state.
        """
        pass
    
    @abstractmethod
    def get_opponents(self, seed: int = 42) -> Dict[str, Any]:
        """Get dictionary of opponent policies for testing.
        
        For multi-player games, this returns policies that can be used
        as opponents during training and evaluation. For single-player
        games, this may return different environment configurations.
        
        Parameters
        ----------
        seed : int, default=42
            Random seed for reproducibility.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary mapping opponent names to policy objects.
            Each policy must have a ``get_action(observation, legal_actions)``
            method that returns an action index.
        """
        pass
