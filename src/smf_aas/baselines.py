"""
Baseline drift detection methods for comparison.

This module implements established changepoint/drift detection algorithms
to compare against the SMF-AAS framework:

1. CUSUM (Cumulative Sum) - Page (1954)
2. Page-Hinkley Test - Page (1954)
3. ADWIN (Adaptive Windowing) - Bifet & Gavaldà (2007)
4. Performance-Only baseline

References
----------
.. [1] Page, E. S. (1954). Continuous inspection schemes.
       Biometrika, 41(1/2), 100-115.
.. [2] Bifet, A., & Gavaldà, R. (2007). Learning from time-changing data
       with adaptive windowing. In SDM (pp. 443-448).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DetectionResult:
    """Result from a detection method.
    
    Attributes
    ----------
    detected : bool
        Whether a change was detected.
    location : Optional[int]
        Index where change was detected.
    score : float
        Detection statistic value.
    method : str
        Name of detection method.
    """
    detected: bool
    location: Optional[int]
    score: float
    method: str


class CUSUM:
    """Cumulative Sum (CUSUM) change detection.
    
    Detects shifts in the mean of a sequence by accumulating
    deviations from a target value. Implements two-sided CUSUM
    that detects both increases and decreases.
    
    Parameters
    ----------
    threshold : float, default=5.0
        Detection threshold (h).
    drift : float, default=0.5
        Allowable drift/slack (k), typically 0.5 * expected_shift.
    warmup : int, default=50
        Episodes before detection starts (for estimating baseline).
    
    References
    ----------
    .. [1] Page, E. S. (1954). Continuous inspection schemes.
           Biometrika, 41(1/2), 100-115.
    
    Examples
    --------
    >>> cusum = CUSUM(threshold=5.0)
    >>> for i, value in enumerate(data):
    ...     result = cusum.update(value)
    ...     if result and result.detected:
    ...         print(f"Change detected at {i}")
    """
    
    def __init__(
        self,
        threshold: float = 5.0,
        drift: float = 0.5,
        warmup: int = 50
    ) -> None:
        self.threshold = threshold
        self.drift = drift
        self.warmup = warmup
        
        self._target: Optional[float] = None
        self._std: float = 1.0
        self._s_pos: float = 0.0  # Upper CUSUM
        self._s_neg: float = 0.0  # Lower CUSUM
        self._n: int = 0
        self._warmup_values: List[float] = []
        self.alerts: List[int] = []
    
    def update(self, value: float) -> Optional[DetectionResult]:
        """Update CUSUM with new value.
        
        Parameters
        ----------
        value : float
            New observation.
        
        Returns
        -------
        Optional[DetectionResult]
            Detection result if change detected.
        """
        self._n += 1
        
        # Warmup period
        if self._target is None:
            self._warmup_values.append(value)
            if len(self._warmup_values) >= self.warmup:
                self._target = float(np.mean(self._warmup_values))
                self._std = max(float(np.std(self._warmup_values)), 0.01)
            return None
        
        # Normalize
        normalized = (value - self._target) / self._std
        
        # Update CUSUM statistics
        self._s_pos = max(0, self._s_pos + normalized - self.drift)
        self._s_neg = max(0, self._s_neg - normalized - self.drift)
        
        # Check for change
        if self._s_pos > self.threshold or self._s_neg > self.threshold:
            self.alerts.append(self._n)
            score = max(self._s_pos, self._s_neg)
            
            # Reset after detection
            self._s_pos = 0.0
            self._s_neg = 0.0
            
            return DetectionResult(
                detected=True,
                location=self._n,
                score=score,
                method="CUSUM"
            )
        
        return None
    
    def reset(self) -> None:
        """Reset detector state."""
        self._target = None
        self._std = 1.0
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._n = 0
        self._warmup_values = []
        self.alerts = []


class PageHinkley:
    """Page-Hinkley test for change detection.
    
    Similar to CUSUM but uses cumulative sum of differences
    from running mean.
    
    Parameters
    ----------
    threshold : float, default=50.0
        Detection threshold (λ).
    alpha : float, default=0.005
        Forgetting factor.
    min_instances : int, default=50
        Minimum observations before detection.
    
    References
    ----------
    .. [1] Page, E. S. (1954). Continuous inspection schemes.
           Biometrika, 41(1/2), 100-115.
    """
    
    def __init__(
        self,
        threshold: float = 50.0,
        alpha: float = 0.005,
        min_instances: int = 50
    ) -> None:
        self.threshold = threshold
        self.alpha = alpha
        self.min_instances = min_instances
        
        self._n: int = 0
        self._sum: float = 0.0
        self._mean: float = 0.0
        self._cumsum: float = 0.0
        self._cumsum_min: float = float('inf')
        self.alerts: List[int] = []
    
    def update(self, value: float) -> Optional[DetectionResult]:
        """Update with new value."""
        self._n += 1
        
        # Update running mean
        self._sum += value
        self._mean = self._sum / self._n
        
        # Update cumulative sum
        self._cumsum += value - self._mean - self.alpha
        self._cumsum_min = min(self._cumsum_min, self._cumsum)
        
        # Page-Hinkley statistic
        ph_stat = self._cumsum - self._cumsum_min
        
        # Check for change
        if self._n >= self.min_instances and ph_stat > self.threshold:
            self.alerts.append(self._n)
            
            result = DetectionResult(
                detected=True,
                location=self._n,
                score=ph_stat,
                method="PageHinkley"
            )
            
            # Reset
            self._cumsum = 0.0
            self._cumsum_min = float('inf')
            
            return result
        
        return None
    
    def reset(self) -> None:
        """Reset detector state."""
        self._n = 0
        self._sum = 0.0
        self._mean = 0.0
        self._cumsum = 0.0
        self._cumsum_min = float('inf')
        self.alerts = []


class ADWIN:
    """Adaptive Windowing (ADWIN) for concept drift detection.
    
    Maintains a variable-length window of recent data and detects
    change by comparing sub-windows using statistical bounds.
    
    Parameters
    ----------
    delta : float, default=0.002
        Confidence parameter (smaller = more sensitive).
    
    References
    ----------
    .. [1] Bifet, A., & Gavaldà, R. (2007). Learning from time-changing
           data with adaptive windowing. In SDM (pp. 443-448).
    """
    
    def __init__(self, delta: float = 0.002) -> None:
        self.delta = delta
        
        self._window: List[float] = []
        self._n: int = 0
        self.alerts: List[int] = []
    
    def update(self, value: float) -> Optional[DetectionResult]:
        """Update ADWIN with new value."""
        self._window.append(value)
        self._n += 1
        
        if len(self._window) < 10:
            return None
        
        # Check for change by comparing sub-windows
        detected, cut_point = self._check_change()
        
        if detected:
            self.alerts.append(self._n)
            
            # Drop old data
            self._window = self._window[cut_point:]
            
            return DetectionResult(
                detected=True,
                location=self._n,
                score=0.0,  # ADWIN doesn't produce a score
                method="ADWIN"
            )
        
        return None
    
    def _check_change(self) -> tuple:
        """Check for change using ADWIN algorithm."""
        n = len(self._window)
        
        for i in range(1, n):
            n1, n2 = i, n - i
            if n1 < 5 or n2 < 5:
                continue
            
            w1 = self._window[:i]
            w2 = self._window[i:]
            
            mean1, mean2 = np.mean(w1), np.mean(w2)
            
            # Compute ADWIN bound
            m = 1.0 / n1 + 1.0 / n2
            delta_prime = self.delta / np.log(n + 1)
            epsilon = np.sqrt(2 * m * np.log(2 / delta_prime))
            
            if abs(mean1 - mean2) > epsilon:
                return True, i
        
        return False, 0
    
    def reset(self) -> None:
        """Reset detector state."""
        self._window = []
        self._n = 0
        self.alerts = []


class PerformanceOnly:
    """Simple baseline: only monitors performance using effect size.
    
    This baseline only tracks performance (returns) without any
    state or behavior monitoring. Used to demonstrate the value
    of multi-component monitoring.
    
    Parameters
    ----------
    window_size : int, default=50
        Window size for comparison.
    baseline_episodes : int, default=200
        Episodes for baseline estimation.
    threshold : float, default=2.5
        Z-score threshold for detection.
    """
    
    def __init__(
        self,
        window_size: int = 50,
        baseline_episodes: int = 200,
        threshold: float = 2.5
    ) -> None:
        self.window_size = window_size
        self.baseline_episodes = baseline_episodes
        self.threshold = threshold
        
        self._returns: List[float] = []
        self._baseline_d_values: List[float] = []
        self._baseline_mean: Optional[float] = None
        self._baseline_std: Optional[float] = None
        self._n: int = 0
        self.alerts: List[int] = []
    
    def update(self, episode_return: float) -> Optional[DetectionResult]:
        """Update with episode return."""
        self._n += 1
        self._returns.append(episode_return)
        
        if len(self._returns) < 2 * self.window_size:
            return None
        
        # Compute Cohen's d between windows
        w1 = self._returns[-2 * self.window_size:-self.window_size]
        w2 = self._returns[-self.window_size:]
        d = self._cohens_d(w1, w2)
        
        # Baseline period
        if self._n <= self.baseline_episodes:
            self._baseline_d_values.append(d)
            if self._n == self.baseline_episodes:
                self._baseline_mean = float(np.mean(self._baseline_d_values))
                self._baseline_std = max(float(np.std(self._baseline_d_values)), 0.01)
            return None
        
        # Detection period
        z = (d - self._baseline_mean) / self._baseline_std
        
        if abs(z) >= self.threshold:
            self.alerts.append(self._n)
            return DetectionResult(
                detected=True,
                location=self._n,
                score=z,
                method="PerformanceOnly"
            )
        
        return None
    
    def _cohens_d(self, w1: List[float], w2: List[float]) -> float:
        """Compute Cohen's d between two windows."""
        if len(w1) < 2 or len(w2) < 2:
            return 0.0
        
        m1, m2 = np.mean(w1), np.mean(w2)
        v1, v2 = np.var(w1, ddof=1), np.var(w2, ddof=1)
        pooled = np.sqrt((v1 + v2) / 2)
        
        return abs(m1 - m2) / max(pooled, 0.01)
    
    def reset(self) -> None:
        """Reset detector state."""
        self._returns = []
        self._baseline_d_values = []
        self._baseline_mean = None
        self._baseline_std = None
        self._n = 0
        self.alerts = []
