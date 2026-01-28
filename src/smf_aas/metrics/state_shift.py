"""
State Distribution Shift Detection.

This module implements the S (State) component of the SMF-AAS framework,
which detects changes in the distribution of states visited by an agent.

The implementation uses Jensen-Shannon divergence computed per-dimension
and averaged, which is more robust than joint-state hashing for continuous
or high-dimensional state spaces.

References
----------
.. [1] Lin, J. (1991). Divergence measures based on the Shannon entropy.
       IEEE Transactions on Information Theory, 37(1), 145-151.
.. [2] Endres, D. M., & Schindelin, J. E. (2003). A new metric for probability
       distributions. IEEE Transactions on Information Theory, 49(7), 1858-1860.

Notes
-----
Jensen-Shannon divergence is a symmetric, bounded (0 to ln(2)) divergence
measure. It is defined as:

    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)

where M = 0.5 * (P + Q) is the average distribution.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Any


class StateDistributionShift:
    """Detects shifts in state distribution using Jensen-Shannon divergence.
    
    This detector computes per-dimension histograms using shared bins between
    consecutive windows and averages the Jensen-Shannon divergence across all
    dimensions. This approach is more robust than joint-state hashing for
    high-cardinality or continuous state spaces.
    
    Parameters
    ----------
    window_size : int, default=50
        Number of episodes per comparison window.
    n_bins : int, default=20
        Number of histogram bins per dimension.
    smoothing : float, default=1e-10
        Laplace smoothing to prevent log(0).
    
    Attributes
    ----------
    window_size : int
        Episodes per window.
    episode_count : int
        Total episodes processed.
    history : List[float]
        History of JS divergence values.
    
    Examples
    --------
    >>> detector = StateDistributionShift(window_size=50)
    >>> for episode in range(100):
    ...     states = collect_episode_states(env)
    ...     js_div = detector.add_episode(states)
    ...     if js_div is not None:
    ...         print(f"JS divergence: {js_div:.4f}")
    
    Notes
    -----
    The detector requires at least 2 full windows to compute divergence.
    Returns None until the second window is complete.
    """
    
    def __init__(
        self,
        window_size: int = 50,
        n_bins: int = 20,
        smoothing: float = 1e-10
    ) -> None:
        self.window_size = window_size
        self.n_bins = n_bins
        self.smoothing = smoothing
        
        self._current_window: List[List[np.ndarray]] = []
        self._previous_window: List[List[np.ndarray]] = []
        self.episode_count: int = 0
        self.history: List[float] = []
    
    def add_episode(self, states: List[Any]) -> Optional[float]:
        """Add episode states and compute divergence if window complete.
        
        Parameters
        ----------
        states : List[Any]
            Sequence of states from the episode. Each state should be
            array-like and will be converted to a flat numpy array.
        
        Returns
        -------
        Optional[float]
            Jensen-Shannon divergence between current and previous windows,
            or None if not enough data yet.
        """
        self.episode_count += 1
        
        # Convert states to flat numpy arrays
        processed_states = [
            np.asarray(s, dtype=np.float64).flatten() for s in states
        ]
        self._current_window.append(processed_states)
        
        # Check if window is complete
        if len(self._current_window) >= self.window_size:
            js_value = None
            
            if len(self._previous_window) >= self.window_size:
                js_value = self._compute_js_divergence(
                    self._current_window, self._previous_window
                )
                self.history.append(js_value)
            
            # Rotate windows
            self._previous_window = self._current_window
            self._current_window = []
            
            return js_value
        
        return None
    
    def _compute_js_divergence(
        self,
        current: List[List[np.ndarray]],
        previous: List[List[np.ndarray]]
    ) -> float:
        """Compute average JS divergence across state dimensions.
        
        Parameters
        ----------
        current : List[List[np.ndarray]]
            Current window episodes, each containing list of state arrays.
        previous : List[List[np.ndarray]]
            Previous window episodes.
        
        Returns
        -------
        float
            Average Jensen-Shannon divergence across dimensions.
        """
        # Flatten all states into arrays
        curr_flat = self._flatten_window(current)
        prev_flat = self._flatten_window(previous)
        
        if curr_flat.size == 0 or prev_flat.size == 0:
            return 0.0
        
        # Ensure 2D (n_observations, n_dimensions)
        if curr_flat.ndim == 1:
            curr_flat = curr_flat.reshape(-1, 1)
        if prev_flat.ndim == 1:
            prev_flat = prev_flat.reshape(-1, 1)
        
        n_dims = min(curr_flat.shape[1], prev_flat.shape[1])
        js_values = []
        
        for dim in range(n_dims):
            curr_vals = curr_flat[:, dim]
            prev_vals = prev_flat[:, dim]
            
            # Remove NaN values (from padding)
            curr_vals = curr_vals[~np.isnan(curr_vals)]
            prev_vals = prev_vals[~np.isnan(prev_vals)]
            
            if len(curr_vals) == 0 or len(prev_vals) == 0:
                js_values.append(0.0)
                continue
            
            js = self._compute_1d_js(curr_vals, prev_vals)
            js_values.append(js)
        
        return float(np.mean(js_values)) if js_values else 0.0
    
    def _flatten_window(self, window: List[List[np.ndarray]]) -> np.ndarray:
        """Flatten window into 2D array with NaN padding."""
        rows = []
        for episode in window:
            for state in episode:
                rows.append(state)
        
        if not rows:
            return np.empty((0,))
        
        # Pad to equal length
        max_len = max(len(r) for r in rows)
        arr = np.full((len(rows), max_len), np.nan, dtype=np.float64)
        for i, r in enumerate(rows):
            arr[i, :len(r)] = r
        
        return arr
    
    def _compute_1d_js(self, p_vals: np.ndarray, q_vals: np.ndarray) -> float:
        """Compute JS divergence for 1D distributions."""
        # Combined range for shared bins
        combined = np.concatenate([p_vals, q_vals])
        mn, mx = np.min(combined), np.max(combined)
        
        if mx - mn < 1e-12:
            return 0.0
        
        # Compute histograms with shared bins
        bins = np.linspace(mn - 1e-12, mx + 1e-12, self.n_bins + 1)
        p_counts, _ = np.histogram(p_vals, bins=bins)
        q_counts, _ = np.histogram(q_vals, bins=bins)
        
        # Add smoothing and normalize
        p_probs = (p_counts.astype(np.float64) + self.smoothing)
        q_probs = (q_counts.astype(np.float64) + self.smoothing)
        p_probs /= p_probs.sum()
        q_probs /= q_probs.sum()
        
        # JS divergence: 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        m_probs = 0.5 * (p_probs + q_probs)
        
        # Compute KL divergences (with numerical stability)
        kl_pm = np.sum(p_probs * np.log(p_probs / m_probs + 1e-12))
        kl_qm = np.sum(q_probs * np.log(q_probs / m_probs + 1e-12))
        
        return 0.5 * kl_pm + 0.5 * kl_qm
    
    def get_history(self) -> List[float]:
        """Get history of divergence values.
        
        Returns
        -------
        List[float]
            Copy of the divergence history.
        """
        return self.history.copy()
    
    def reset(self) -> None:
        """Reset detector state."""
        self._current_window = []
        self._previous_window = []
        self.episode_count = 0
        self.history = []
