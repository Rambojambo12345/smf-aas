"""
Behavioral Shift Detection.

This module implements the B (Behavior) component of the SMF-AAS framework,
which detects changes in agent behavior through a multi-dimensional feature
space capturing action patterns, temporal dynamics, and exploration behavior.

The behavioral feature vector includes:
- Action entropy (normalized)
- Mean episode length
- State revisitation rate
- Action persistence (consecutive same actions)
- Return variance

Changes are detected via Euclidean distance in normalized feature space.

Notes
-----
Unlike state distribution which captures "where" the agent goes, behavioral
features capture "how" the agent behaves - its decision patterns and dynamics.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Any
from collections import Counter


@dataclass(frozen=True)
class BehaviorFeatures:
    """Immutable container for behavioral feature vector.
    
    Attributes
    ----------
    action_entropy : float
        Normalized entropy of action distribution (0 to 1).
    episode_length : float
        Mean number of actions per episode.
    state_revisitation : float
        Fraction of states that are revisits (0 to 1).
    action_persistence : float
        Fraction of consecutive same-action pairs (0 to 1).
    return_variance : float
        Variance of episode returns.
    """
    action_entropy: float
    episode_length: float
    state_revisitation: float
    action_persistence: float
    return_variance: float
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy array for distance computation."""
        return np.array([
            self.action_entropy,
            self.episode_length,
            self.state_revisitation,
            self.action_persistence,
            self.return_variance
        ], dtype=np.float64)
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get ordered list of feature names."""
        return [
            "action_entropy",
            "episode_length", 
            "state_revisitation",
            "action_persistence",
            "return_variance"
        ]


@dataclass
class EpisodeData:
    """Container for single episode trajectory data.
    
    Parameters
    ----------
    states : List[Any]
        Sequence of states visited.
    actions : List[int]
        Sequence of actions taken.
    rewards : List[float]
        Sequence of rewards received.
    """
    states: List[Any]
    actions: List[int]
    rewards: List[float]
    
    @property
    def episode_return(self) -> float:
        """Total episode return."""
        return sum(self.rewards) if self.rewards else 0.0


class BehaviorShiftDetector:
    """Detects behavioral shifts using feature-space distance.
    
    This detector extracts a multi-dimensional behavioral feature vector
    from each window of episodes and computes the Euclidean distance between
    consecutive windows in normalized feature space.
    
    Parameters
    ----------
    window_size : int, default=50
        Number of episodes per comparison window.
    n_actions : int, default=9
        Number of possible actions (for entropy normalization).
    
    Attributes
    ----------
    window_size : int
        Episodes per window.
    n_actions : int
        Number of possible actions.
    episode_count : int
        Total episodes processed.
    history : List[float]
        History of feature-space distances.
    feature_history : List[BehaviorFeatures]
        History of computed feature vectors.
    
    Examples
    --------
    >>> detector = BehaviorShiftDetector(window_size=50, n_actions=9)
    >>> for episode in range(100):
    ...     states, actions, rewards = collect_episode(env)
    ...     distance = detector.add_episode(states, actions, rewards)
    ...     if distance is not None:
    ...         print(f"Behavioral distance: {distance:.4f}")
    """
    
    def __init__(self, window_size: int = 50, n_actions: int = 9) -> None:
        self.window_size = window_size
        self.n_actions = n_actions
        
        self._current_window: List[EpisodeData] = []
        self._previous_window: List[EpisodeData] = []
        self._previous_features: Optional[BehaviorFeatures] = None
        
        self.episode_count: int = 0
        self.history: List[float] = []
        self.feature_history: List[BehaviorFeatures] = []
    
    def add_episode(
        self,
        states: List[Any],
        actions: List[int],
        rewards: List[float]
    ) -> Optional[float]:
        """Add episode data and compute distance if window complete.
        
        Parameters
        ----------
        states : List[Any]
            Sequence of states visited.
        actions : List[int]
            Sequence of actions taken.
        rewards : List[float]
            Sequence of rewards received.
        
        Returns
        -------
        Optional[float]
            Feature-space distance between windows, or None if not ready.
        """
        self.episode_count += 1
        
        episode = EpisodeData(states=states, actions=actions, rewards=rewards)
        self._current_window.append(episode)
        
        if len(self._current_window) >= self.window_size:
            current_features = self._compute_features(self._current_window)
            self.feature_history.append(current_features)
            
            distance = None
            if self._previous_features is not None:
                distance = self._compute_distance(
                    current_features, self._previous_features
                )
                self.history.append(distance)
            
            # Rotate windows
            self._previous_window = self._current_window
            self._previous_features = current_features
            self._current_window = []
            
            return distance
        
        return None
    
    def _compute_features(self, episodes: List[EpisodeData]) -> BehaviorFeatures:
        """Extract behavioral features from episode window."""
        return BehaviorFeatures(
            action_entropy=self._compute_action_entropy(episodes),
            episode_length=self._compute_episode_length(episodes),
            state_revisitation=self._compute_state_revisitation(episodes),
            action_persistence=self._compute_action_persistence(episodes),
            return_variance=self._compute_return_variance(episodes)
        )
    
    def _compute_action_entropy(self, episodes: List[EpisodeData]) -> float:
        """Compute normalized action distribution entropy."""
        all_actions = []
        for ep in episodes:
            all_actions.extend(ep.actions)
        
        if not all_actions:
            return 0.0
        
        counts = Counter(all_actions)
        total = len(all_actions)
        
        # Shannon entropy
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log(p + 1e-12)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(self.n_actions)
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _compute_episode_length(self, episodes: List[EpisodeData]) -> float:
        """Compute mean episode length (number of actions)."""
        if not episodes:
            return 0.0
        lengths = [len(ep.actions) for ep in episodes]
        return float(np.mean(lengths))
    
    def _compute_state_revisitation(self, episodes: List[EpisodeData]) -> float:
        """Compute mean state revisitation rate."""
        if not episodes:
            return 0.0
        
        rates = []
        for ep in episodes:
            if len(ep.states) <= 1:
                rates.append(0.0)
                continue
            
            # Hash states for comparison
            state_hashes = [
                tuple(np.asarray(s).flatten().round(6)) for s in ep.states
            ]
            n_unique = len(set(state_hashes))
            n_total = len(state_hashes)
            
            # Revisitation rate: fraction that are revisits
            revisit_rate = 1.0 - (n_unique / n_total)
            rates.append(revisit_rate)
        
        return float(np.mean(rates))
    
    def _compute_action_persistence(self, episodes: List[EpisodeData]) -> float:
        """Compute mean action persistence (consecutive same actions)."""
        if not episodes:
            return 0.0
        
        rates = []
        for ep in episodes:
            if len(ep.actions) <= 1:
                rates.append(0.0)
                continue
            
            same_count = sum(
                1 for i in range(1, len(ep.actions))
                if ep.actions[i] == ep.actions[i - 1]
            )
            persistence = same_count / (len(ep.actions) - 1)
            rates.append(persistence)
        
        return float(np.mean(rates))
    
    def _compute_return_variance(self, episodes: List[EpisodeData]) -> float:
        """Compute variance of episode returns."""
        if len(episodes) < 2:
            return 0.0
        
        returns = [ep.episode_return for ep in episodes]
        return float(np.var(returns, ddof=1))
    
    def _compute_distance(
        self,
        current: BehaviorFeatures,
        previous: BehaviorFeatures
    ) -> float:
        """Compute normalized Euclidean distance between feature vectors."""
        curr_vec = current.to_vector()
        prev_vec = previous.to_vector()
        
        # Normalize by scale to make features comparable
        # Use max of absolute values with minimum to prevent division issues
        scale = np.maximum(np.abs(prev_vec), np.abs(curr_vec))
        scale = np.maximum(scale, 0.1)  # Minimum scale
        
        normalized_diff = (curr_vec - prev_vec) / scale
        
        # Clip to prevent extreme values
        normalized_diff = np.clip(normalized_diff, -10, 10)
        
        return float(np.linalg.norm(normalized_diff))
    
    def get_history(self) -> List[float]:
        """Get history of distance values."""
        return self.history.copy()
    
    def get_feature_history(self) -> List[BehaviorFeatures]:
        """Get history of feature vectors."""
        return self.feature_history.copy()
    
    def reset(self) -> None:
        """Reset detector state."""
        self._current_window = []
        self._previous_window = []
        self._previous_features = None
        self.episode_count = 0
        self.history = []
        self.feature_history = []
