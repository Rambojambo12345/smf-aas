"""
Baseline drift detection methods for comparison.

This module implements established changepoint/drift detection algorithms
to compare against the SMF-AAS framework:

1. CUSUM (Cumulative Sum) - Page (1954)
2. Page-Hinkley Test - Page (1954)
3. ADWIN (Adaptive Windowing) - Bifet & Gavaldà (2007)
4. DDM (Drift Detection Method) - Gama et al. (2004)
5. EDDM (Early Drift Detection Method) - Baena-García et al. (2006)
6. KSWIN (Kolmogorov-Smirnov Windowing) - Raab et al. (2020)
7. Performance-Only baseline

"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from scipy import stats


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
        self._s_pos: float = 0.0
        self._s_neg: float = 0.0
        self._n: int = 0
        self._warmup_values: List[float] = []
        self.alerts: List[int] = []
    
    def update(self, value: float) -> Optional[DetectionResult]:
        """Update CUSUM with new value."""
        self._n += 1
        
        if self._target is None:
            self._warmup_values.append(value)
            if len(self._warmup_values) >= self.warmup:
                self._target = float(np.mean(self._warmup_values))
                self._std = max(float(np.std(self._warmup_values)), 0.01)
            return None
        
        normalized = (value - self._target) / self._std
        
        self._s_pos = max(0, self._s_pos + normalized - self.drift)
        self._s_neg = max(0, self._s_neg - normalized - self.drift)
        
        if self._s_pos > self.threshold or self._s_neg > self.threshold:
            self.alerts.append(self._n)
            score = max(self._s_pos, self._s_neg)
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
        
        self._sum += value
        self._mean = self._sum / self._n
        
        self._cumsum += value - self._mean - self.alpha
        self._cumsum_min = min(self._cumsum_min, self._cumsum)
        
        ph_stat = self._cumsum - self._cumsum_min
        
        if self._n >= self.min_instances and ph_stat > self.threshold:
            self.alerts.append(self._n)
            
            result = DetectionResult(
                detected=True,
                location=self._n,
                score=ph_stat,
                method="PageHinkley"
            )
            
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
        
        detected, cut_point = self._check_change()
        
        if detected:
            self.alerts.append(self._n)
            self._window = self._window[cut_point:]
            
            return DetectionResult(
                detected=True,
                location=self._n,
                score=0.0,
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


class DDM:
    """Drift Detection Method (DDM).
    
    Monitors the error rate and its standard deviation. Detects drift
    when the error rate significantly increases beyond a threshold.
    
    Parameters
    ----------
    warning_level : float, default=2.0
        Number of standard deviations for warning.
    drift_level : float, default=3.0
        Number of standard deviations for drift detection.
    min_instances : int, default=30
        Minimum observations before detection.
    
    References
    ----------
    .. [1] Gama, J., Medas, P., Castillo, G., & Rodrigues, P. (2004). 
           Learning with drift detection. In SBIA (pp. 286-295).
    
    Notes
    -----
    DDM was originally designed for classification error. Here we adapt it
    for continuous values by treating values below the running mean as 
    "errors" (performance drops).
    """
    
    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_instances: int = 30
    ) -> None:
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_instances = min_instances
        
        self._n: int = 0
        self._mean: float = 0.0
        self._std: float = 0.0
        self._p_min: float = float('inf')
        self._s_min: float = float('inf')
        self._in_warning: bool = False
        self._running_mean: float = 0.0
        self.alerts: List[int] = []
    
    def update(self, value: float) -> Optional[DetectionResult]:
        """Update DDM with new value."""
        self._n += 1
        
        # Running mean of the raw values
        if self._n == 1:
            self._running_mean = value
        else:
            self._running_mean += (value - self._running_mean) / self._n
        
        # Binary error: 1 if below running mean (performance drop), 0 otherwise
        error = 1.0 if value < self._running_mean else 0.0
        
        # Update error rate statistics
        self._mean += (error - self._mean) / self._n
        self._std = np.sqrt(self._mean * (1 - self._mean) / self._n) if self._n > 1 else 0.0
        
        if self._n < self.min_instances:
            return None
        
        # Track minimum error rate + std
        if self._mean + self._std < self._p_min + self._s_min:
            self._p_min = self._mean
            self._s_min = self._std
        
        # Check for drift
        if self._s_min > 0 and self._mean + self._std >= self._p_min + self.drift_level * self._s_min:
            self.alerts.append(self._n)
            self._reset_stats()
            return DetectionResult(
                detected=True,
                location=self._n,
                score=self._mean + self._std,
                method="DDM"
            )
        
        return None
    
    def _reset_stats(self) -> None:
        """Reset statistics after drift detection."""
        self._mean = 0.0
        self._std = 0.0
        self._p_min = float('inf')
        self._s_min = float('inf')
        self._in_warning = False
    
    def reset(self) -> None:
        """Reset detector state."""
        self._n = 0
        self._mean = 0.0
        self._std = 0.0
        self._p_min = float('inf')
        self._s_min = float('inf')
        self._in_warning = False
        self._running_mean = 0.0
        self.alerts = []


class EDDM:
    """Early Drift Detection Method (EDDM).
    
    Extension of DDM that monitors the distance between classification
    errors rather than just the error rate. More sensitive to gradual drift.
    
    Parameters
    ----------
    warning_level : float, default=0.95
        Threshold ratio for warning level.
    drift_level : float, default=0.90
        Threshold ratio for drift detection.
    min_instances : int, default=30
        Minimum observations before detection.
    
    References
    ----------
    .. [1] Baena-García, M., et al. (2006). Early drift detection method.
           In ECML PKDD Workshop on Knowledge Discovery from Data Streams.
    """
    
    def __init__(
        self,
        warning_level: float = 0.95,
        drift_level: float = 0.90,
        min_instances: int = 30
    ) -> None:
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_instances = min_instances
        
        self._n: int = 0
        self._num_errors: int = 0
        self._last_error: int = 0
        self._mean_distance: float = 0.0
        self._std_distance: float = 0.0
        self._m2: float = 0.0
        self._max_distance: float = 0.0
        self._max_std: float = 0.0
        self._running_mean: float = 0.0
        self.alerts: List[int] = []
    
    def update(self, value: float) -> Optional[DetectionResult]:
        """Update EDDM with new value."""
        self._n += 1
        
        # Running mean
        if self._n == 1:
            self._running_mean = value
        else:
            self._running_mean += (value - self._running_mean) / self._n
        
        # Check if this is an "error" (below running mean)
        is_error = value < self._running_mean
        
        if is_error:
            self._num_errors += 1
            
            if self._num_errors > 1:
                distance = self._n - self._last_error
                
                # Welford's online algorithm for mean and variance
                delta = distance - self._mean_distance
                self._mean_distance += delta / self._num_errors
                delta2 = distance - self._mean_distance
                self._m2 += delta * delta2
                
                if self._num_errors > 2:
                    self._std_distance = np.sqrt(self._m2 / (self._num_errors - 1))
            
            self._last_error = self._n
        
        if self._n < self.min_instances or self._num_errors < 2:
            return None
        
        # Track maximum (mean + 2*std)
        current_score = self._mean_distance + 2 * self._std_distance
        if current_score > self._max_distance + 2 * self._max_std:
            self._max_distance = self._mean_distance
            self._max_std = self._std_distance
        
        max_score = self._max_distance + 2 * self._max_std
        
        if max_score > 0:
            ratio = current_score / max_score
            
            if ratio < self.drift_level:
                self.alerts.append(self._n)
                self._reset_stats()
                return DetectionResult(
                    detected=True,
                    location=self._n,
                    score=ratio,
                    method="EDDM"
                )
        
        return None
    
    def _reset_stats(self) -> None:
        """Reset statistics after drift detection."""
        self._num_errors = 0
        self._last_error = 0
        self._mean_distance = 0.0
        self._std_distance = 0.0
        self._m2 = 0.0
        self._max_distance = 0.0
        self._max_std = 0.0
    
    def reset(self) -> None:
        """Reset detector state."""
        self._n = 0
        self._num_errors = 0
        self._last_error = 0
        self._mean_distance = 0.0
        self._std_distance = 0.0
        self._m2 = 0.0
        self._max_distance = 0.0
        self._max_std = 0.0
        self._running_mean = 0.0
        self.alerts = []


class KSWIN:
    """Kolmogorov-Smirnov Windowing (KSWIN) for drift detection.
    
    Uses the Kolmogorov-Smirnov test to compare distributions in
    a reference window and a sliding window.
    
    Parameters
    ----------
    alpha : float, default=0.005
        Significance level for the KS test.
    window_size : int, default=100
        Size of the sliding window.
    stat_size : int, default=30
        Size of the reference (stationary) window.
    
    References
    ----------
    .. [1] Raab, C., Heusinger, M., & Schleif, F. M. (2020). Reactive soft 
           prototype computing for concept drift streams. Neurocomputing.
    """
    
    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30
    ) -> None:
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        
        self._window: List[float] = []
        self._n: int = 0
        self.alerts: List[int] = []
    
    def update(self, value: float) -> Optional[DetectionResult]:
        """Update KSWIN with new value."""
        self._n += 1
        self._window.append(value)
        
        # Keep window bounded
        if len(self._window) > self.window_size:
            self._window.pop(0)
        
        # Need enough data for both windows
        if len(self._window) < self.stat_size + self.stat_size:
            return None
        
        # Reference window (older data) vs recent window
        ref_window = self._window[:self.stat_size]
        recent_window = self._window[-self.stat_size:]
        
        # Perform KS test
        ks_stat, p_value = stats.ks_2samp(ref_window, recent_window)
        
        if p_value < self.alpha:
            self.alerts.append(self._n)
            # Reset window after detection
            self._window = self._window[-self.stat_size:]
            
            return DetectionResult(
                detected=True,
                location=self._n,
                score=ks_stat,
                method="KSWIN"
            )
        
        return None
    
    def reset(self) -> None:
        """Reset detector state."""
        self._window = []
        self._n = 0
        self.alerts = []


class PerformanceOnly:
    """Simple baseline: only monitors performance using effect size.
    
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
        
        w1 = self._returns[-2 * self.window_size:-self.window_size]
        w2 = self._returns[-self.window_size:]
        d = self._cohens_d(w1, w2)
        
        if self._n <= self.baseline_episodes:
            self._baseline_d_values.append(d)
            if self._n == self.baseline_episodes:
                self._baseline_mean = float(np.mean(self._baseline_d_values))
                self._baseline_std = max(float(np.std(self._baseline_d_values)), 0.01)
            return None
        
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
