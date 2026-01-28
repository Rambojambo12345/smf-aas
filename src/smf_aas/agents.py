"""
Agent implementations for experiments.

This module provides learning agents for experimental evaluation:
- TabularQLearning: Tabular Q-learning with ε-greedy exploration
- RandomAgent: Baseline random policy

References
----------
.. [1] Watkins, C. J., & Dayan, P. (1992). Q-learning.
       Machine learning, 8(3-4), 279-292.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict


class TabularQLearning:
    """Tabular Q-learning agent with ε-greedy exploration.
    
    Uses state hashing for tabular representation, suitable for
    discrete or discretized state spaces.
    
    Parameters
    ----------
    n_actions : int
        Number of possible actions.
    learning_rate : float, default=0.1
        Q-value learning rate (α).
    discount : float, default=0.99
        Discount factor (γ).
    epsilon : float, default=0.1
        Exploration rate for ε-greedy.
    seed : int, optional
        Random seed for reproducibility.
    
    Attributes
    ----------
    q_table : Dict
        Q-values indexed by (state_hash, action).
    n_actions : int
        Number of actions.
    
    Examples
    --------
    >>> agent = TabularQLearning(n_actions=9, epsilon=0.1)
    >>> action = agent.get_action(observation, legal_actions)
    >>> agent.update(reward, next_obs, next_legal, done)
    """
    
    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 0.1,
        seed: Optional[int] = None
    ) -> None:
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        
        self._rng = np.random.RandomState(seed)
        self._q_table: Dict[Tuple, float] = defaultdict(float)
        
        # Track last state-action for updates
        self._last_state: Optional[Tuple] = None
        self._last_action: Optional[int] = None
    
    def _hash_state(self, observation: np.ndarray) -> Tuple:
        """Convert observation to hashable state key."""
        # Round to reduce state space for continuous observations
        return tuple(np.round(observation.flatten(), 3))
    
    def get_action(
        self,
        observation: np.ndarray,
        legal_actions: List[int]
    ) -> int:
        """Select action using ε-greedy policy.
        
        Parameters
        ----------
        observation : np.ndarray
            Current state observation.
        legal_actions : List[int]
            List of legal action indices.
        
        Returns
        -------
        int
            Selected action index.
        """
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        state = self._hash_state(observation)
        
        # ε-greedy action selection
        if self._rng.random() < self.epsilon:
            action = int(self._rng.choice(legal_actions))
        else:
            # Greedy: select best legal action
            q_values = {a: self._q_table[(state, a)] for a in legal_actions}
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = int(self._rng.choice(best_actions))
        
        # Store for update
        self._last_state = state
        self._last_action = action
        
        return action
    
    def update(
        self,
        reward: float,
        next_observation: np.ndarray,
        next_legal_actions: List[int],
        done: bool
    ) -> None:
        """Update Q-value using TD(0).
        
        Parameters
        ----------
        reward : float
            Reward received.
        next_observation : np.ndarray
            Next state observation.
        next_legal_actions : List[int]
            Legal actions in next state.
        done : bool
            Whether episode is finished.
        """
        if self._last_state is None:
            return
        
        state = self._last_state
        action = self._last_action
        next_state = self._hash_state(next_observation)
        
        # Compute TD target
        if done or not next_legal_actions:
            target = reward
        else:
            next_q_values = [self._q_table[(next_state, a)] for a in next_legal_actions]
            target = reward + self.discount * max(next_q_values)
        
        # Update Q-value
        old_q = self._q_table[(state, action)]
        self._q_table[(state, action)] = old_q + self.learning_rate * (target - old_q)
    
    def reset(self) -> None:
        """Reset episode state (not Q-values)."""
        self._last_state = None
        self._last_action = None
    
    def get_q_table_size(self) -> int:
        """Get number of entries in Q-table."""
        return len(self._q_table)


class RandomAgent:
    """Uniform random policy baseline.
    
    Parameters
    ----------
    n_actions : int, optional
        Number of actions (unused, for API compatibility).
    seed : int, optional
        Random seed.
    """
    
    def __init__(self, n_actions: int = None, seed: Optional[int] = None) -> None:
        self._rng = np.random.RandomState(seed)
    
    def get_action(
        self,
        observation: np.ndarray,
        legal_actions: List[int]
    ) -> int:
        """Select random legal action."""
        return int(self._rng.choice(legal_actions))
    
    def update(self, *args: Any, **kwargs: Any) -> None:
        """No-op for compatibility."""
        pass
    
    def reset(self) -> None:
        """No-op for compatibility."""
        pass
