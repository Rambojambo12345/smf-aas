"""
Performance Change Detection.

This module implements the P (Performance) component of the SMF-AAS framework,
which detects changes in agent performance using Cohen's d effect size.

Cohen's d measures the standardized difference between two means:
    d = (M1 - M2) / S_pooled

where S_pooled is the pooled standard deviation.

References
----------
.. [1] Cohen, J. (1988). Statistical power analysis for the behavioral sciences
       (2nd ed.). Lawrence Erlbaum Associates.
.. [2] Sawilowsky, S. S. (2009). New effect size rules of thumb.
       Journal of Modern Applied Statistical Methods, 8(2), 597-599.

Notes
-----
Cohen's d interpretation guidelines:
- |d| < 0.2: negligible
- 0.2 <= |d| < 0.5: small
- 0.5 <= |d| < 0.8: medium
- |d| >= 0.8: large
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


class PerformanceMonitor:
    """Detects performance changes using Cohen's d effect size.
    
    This monitor computes Cohen's d between consecutive windows of episode
    returns, providing a standardized measure of performance change that
    accounts for natural variance.
    
    Parameters
    ----------
    window_size : int, default=50
        Number of episodes per comparison window.
    
    Attributes
    ----------
    window_size : int
        Episodes per window.
    episode_count : int
        Total episodes processed.
    history : List[float]
        History of Cohen's d values.
    returns : List[float]
        All episode returns (for diagnostics).
    
    Examples
    --------
    >>> monitor = PerformanceMonitor(window_size=50)
    >>> for episode in range(100):
    ...     episode_return = run_episode(env, agent)
    ...     cohens_d = monitor.add_episode(episode_return)
    ...     if cohens_d is not None:
    ...         print(f"Cohen's d: {cohens_d:.4f}")
    """
    
    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        
        self._current_window: List[float] = []
        self._previous_window: List[float] = []
        
        self.episode_count: int = 0
        self.history: List[float] = []
        self.returns: List[float] = []
    
    def add_episode(self, episode_return: float) -> Optional[float]:
        """Add episode return and compute effect size if window complete.
        
        Parameters
        ----------
        episode_return : float
            Total return from the episode.
        
        Returns
        -------
        Optional[float]
            Cohen's d between current and previous windows,
            or None if not enough data.
        """
        self.episode_count += 1
        self.returns.append(episode_return)
        self._current_window.append(episode_return)
        
        if len(self._current_window) >= self.window_size:
            cohens_d = None
            
            if len(self._previous_window) >= self.window_size:
                cohens_d = self._compute_cohens_d(
                    self._current_window, self._previous_window
                )
                self.history.append(cohens_d)
            
            # Rotate windows
            self._previous_window = self._current_window.copy()
            self._current_window = []
            
            return cohens_d
        
        return None
    
    def _compute_cohens_d(
        self,
        group1: List[float],
        group2: List[float]
    ) -> float:
        """Compute Cohen's d effect size between two groups.
        
        Uses pooled standard deviation for the denominator.
        
        Parameters
        ----------
        group1 : List[float]
            First group of values (current window).
        group2 : List[float]
            Second group of values (previous window).
        
        Returns
        -------
        float
            Cohen's d effect size. Positive if group1 > group2.
        """
        arr1 = np.array(group1, dtype=np.float64)
        arr2 = np.array(group2, dtype=np.float64)
        
        n1, n2 = len(arr1), len(arr2)
        
        if n1 < 2 or n2 < 2:
            return 0.0
        
        mean1, mean2 = np.mean(arr1), np.mean(arr2)
        var1, var2 = np.var(arr1, ddof=1), np.var(arr2, ddof=1)
        
        # Pooled standard deviation
        # Using the formula: sqrt((var1 + var2) / 2)
        # This is appropriate when sample sizes are equal
        pooled_std = np.sqrt((var1 + var2) / 2)
        
        # Prevent division by zero
        if pooled_std < 1e-10:
            return 0.0
        
        return float((mean1 - mean2) / pooled_std)
    
    def get_window_stats(self) -> dict:
        """Get statistics for current and previous windows.
        
        Returns
        -------
        dict
            Dictionary with mean and std for each window.
        """
        stats = {}
        
        if self._current_window:
            arr = np.array(self._current_window)
            stats["current"] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "n": len(arr)
            }
        
        if self._previous_window:
            arr = np.array(self._previous_window)
            stats["previous"] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "n": len(arr)
            }
        
        return stats
    
    def get_history(self) -> List[float]:
        """Get history of Cohen's d values."""
        return self.history.copy()
    
    def reset(self) -> None:
        """Reset monitor state."""
        self._current_window = []
        self._previous_window = []
        self.episode_count = 0
        self.history = []
        self.returns = []
