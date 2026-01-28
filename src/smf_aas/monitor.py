"""
Strategy Monitor - Core monitoring framework.

This module implements the main SMF-AAS monitoring framework that combines
three components to detect strategy changes in adaptive AI systems:

- **S (State)**: Detects shifts in state distribution using Jensen-Shannon divergence
- **B (Behavior)**: Detects behavioral changes via feature-space distance
- **P (Performance)**: Detects performance changes using Cohen's d effect size

The Composite Drift Score (CDS) aggregates these components:
    CDS = mean(|z_S|, |z_B|, |z_P|)

where z_X = (X - μ_baseline) / σ_baseline

References
----------
.. [1] Lin, J. (1991). Divergence measures based on the Shannon entropy.
       IEEE Transactions on Information Theory, 37(1), 145-151.
.. [2] Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
       Lawrence Erlbaum Associates.

Example
-------
>>> from smf_aas import StrategyMonitor, MonitorConfig
>>> config = MonitorConfig(window_size=50, baseline_episodes=200)
>>> monitor = StrategyMonitor(config, n_actions=9)
>>> 
>>> for episode in range(1500):
...     states, actions, rewards, ret = run_episode(env, agent)
...     alert = monitor.update(states, actions, rewards, ret)
...     if alert:
...         print(f"Alert at episode {episode}: {alert.level.value}")
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from .metrics import (
    StateDistributionShift,
    BehaviorShiftDetector, 
    PerformanceMonitor,
)


class AlertLevel(Enum):
    """Alert severity levels for governance response.
    
    Attributes
    ----------
    GREEN : str
        Normal operation, no action required.
    YELLOW : str
        Warning level, enhanced monitoring recommended.
    RED : str
        Critical level, governance review required.
    """
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class MonitorConfig:
    """Configuration for the Strategy Monitor.
    
    Parameters
    ----------
    window_size : int, default=50
        Number of episodes per comparison window.
    baseline_episodes : int, default=200
        Number of episodes to establish baseline statistics.
    yellow_threshold : float, default=2.0
        CDS threshold for yellow (warning) alerts in standard deviations.
    red_threshold : float, default=3.0
        CDS threshold for red (critical) alerts in standard deviations.
    component_yellow : float, default=2.5
        Individual component threshold for yellow alerts.
    component_red : float, default=3.5
        Individual component threshold for red alerts.
    min_std : float, default=0.01
        Minimum standard deviation to prevent division by zero.
    use_sliding_window : bool, default=False
        If True, use sliding windows with stride for faster detection.
    stride : int, default=10
        Episodes between checks when using sliding windows.
        
    Notes
    -----
    The baseline period should be long enough to capture natural variance
    but short enough to be practical. A general guideline is:
    baseline_episodes >= 4 * window_size
    
    Examples
    --------
    >>> config = MonitorConfig(window_size=50, baseline_episodes=200)
    >>> config.yellow_threshold
    2.0
    """
    window_size: int = 50
    baseline_episodes: int = 200
    yellow_threshold: float = 2.0
    red_threshold: float = 3.0
    component_yellow: float = 2.5
    component_red: float = 3.5
    min_std: float = 0.01
    use_sliding_window: bool = False
    stride: int = 10
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.window_size < 10:
            raise ValueError(f"window_size must be >= 10, got {self.window_size}")
        if self.baseline_episodes < 2 * self.window_size:
            raise ValueError(
                f"baseline_episodes ({self.baseline_episodes}) should be at least "
                f"2 * window_size ({2 * self.window_size})"
            )
        if self.yellow_threshold >= self.red_threshold:
            raise ValueError("yellow_threshold must be less than red_threshold")
        if self.stride > self.window_size:
            raise ValueError("stride should not exceed window_size")


@dataclass
class Alert:
    """Represents a detected strategy change alert.
    
    Attributes
    ----------
    level : AlertLevel
        Severity of the alert (GREEN, YELLOW, RED).
    composite_score : float
        The Composite Drift Score (CDS) that triggered the alert.
    component_scores : Dict[str, float]
        Individual z-scores for each component (S, B, P).
    triggered_by : List[str]
        List of component names that exceeded thresholds.
    episode : int
        Episode number when alert was triggered.
    message : str
        Human-readable alert message.
    """
    level: AlertLevel
    composite_score: float
    component_scores: Dict[str, float]
    triggered_by: List[str]
    episode: int
    message: str = ""
    
    def __post_init__(self) -> None:
        if not self.message:
            level_str = "CRITICAL" if self.level == AlertLevel.RED else "WARNING"
            self.message = (
                f"{level_str}: Strategy change detected at episode {self.episode} "
                f"(CDS={self.composite_score:.2f}σ, triggered by: {', '.join(self.triggered_by)})"
            )


@dataclass
class MonitoringSnapshot:
    """Snapshot of monitoring state at a given episode.
    
    Attributes
    ----------
    episode : int
        Episode number.
    state_shift : Optional[float]
        Raw S component value (Jensen-Shannon divergence).
    behavior_shift : Optional[float]
        Raw B component value (feature-space distance).
    performance_change : Optional[float]
        Raw P component value (Cohen's d).
    z_scores : Dict[str, float]
        Normalized z-scores for each component.
    composite_score : Optional[float]
        Composite Drift Score (CDS).
    alert_level : AlertLevel
        Current alert level.
    """
    episode: int
    state_shift: Optional[float] = None
    behavior_shift: Optional[float] = None
    performance_change: Optional[float] = None
    z_scores: Dict[str, float] = field(default_factory=dict)
    composite_score: Optional[float] = None
    alert_level: AlertLevel = AlertLevel.GREEN


class StrategyMonitor:
    """Main monitoring framework for detecting strategy changes.
    
    The Strategy Monitor tracks three complementary metrics to detect
    when an AI agent's behavior has changed significantly:
    
    1. **State Distribution (S)**: Uses Jensen-Shannon divergence to detect
       changes in which states the agent visits.
    
    2. **Behavioral Features (B)**: Uses Euclidean distance in a feature
       space (action entropy, episode length, etc.) to detect behavioral changes.
    
    3. **Performance (P)**: Uses Cohen's d effect size to detect changes
       in episode returns.
    
    These are combined into a Composite Drift Score (CDS) using z-score
    normalization against a baseline period.
    
    Parameters
    ----------
    config : MonitorConfig
        Configuration parameters.
    n_actions : int
        Number of possible actions in the environment.
    
    Attributes
    ----------
    config : MonitorConfig
        Configuration parameters.
    baseline_complete : bool
        Whether baseline statistics have been established.
    baseline_stats : Dict[str, Tuple[float, float]]
        Baseline (mean, std) for each component.
    history : List[MonitoringSnapshot]
        History of monitoring snapshots.
    alerts : List[Alert]
        List of all alerts generated.
    episode_count : int
        Total episodes processed.
    
    Examples
    --------
    >>> config = MonitorConfig(window_size=50, baseline_episodes=200)
    >>> monitor = StrategyMonitor(config, n_actions=9)
    >>> 
    >>> # Baseline period
    >>> for ep in range(200):
    ...     states, actions, rewards, ret = run_episode(env, agent, opponent_a)
    ...     monitor.update(states, actions, rewards, ret)
    >>> 
    >>> # Detection period (opponent changed)
    >>> for ep in range(200, 400):
    ...     states, actions, rewards, ret = run_episode(env, agent, opponent_b)
    ...     alert = monitor.update(states, actions, rewards, ret)
    ...     if alert and alert.level == AlertLevel.RED:
    ...         print(f"Strategy change detected at episode {ep}")
    
    Notes
    -----
    The monitor requires a baseline period to establish normal operating
    statistics. Alerts are only generated after baseline_episodes have
    been processed.
    
    See Also
    --------
    MonitorConfig : Configuration dataclass
    Alert : Alert dataclass
    """
    
    def __init__(self, config: MonitorConfig, n_actions: int) -> None:
        self.config = config
        self.n_actions = n_actions
        
        # Initialize component detectors
        self._state_detector = StateDistributionShift(
            window_size=config.window_size
        )
        self._behavior_detector = BehaviorShiftDetector(
            window_size=config.window_size,
            n_actions=n_actions
        )
        self._performance_monitor = PerformanceMonitor(
            window_size=config.window_size
        )
        
        # Baseline tracking
        self.baseline_complete: bool = False
        self.baseline_stats: Dict[str, Tuple[float, float]] = {}
        self._baseline_values: Dict[str, List[float]] = {"S": [], "B": [], "P": []}
        
        # Current values
        self._latest: Dict[str, Optional[float]] = {"S": None, "B": None, "P": None}
        
        # History and alerts
        self.history: List[MonitoringSnapshot] = []
        self.alerts: List[Alert] = []
        self.episode_count: int = 0
        
        # Sliding window state
        self._last_check_episode: int = 0
        
        # Diagnostics
        self._std_clamp_count: int = 0
    
    def update(
        self,
        states: List[Any],
        actions: List[int],
        rewards: List[float],
        episode_return: float
    ) -> Optional[Alert]:
        """Update monitor with episode data.
        
        Parameters
        ----------
        states : List[Any]
            Sequence of states visited during the episode.
            Each state should be array-like.
        actions : List[int]
            Sequence of actions taken.
        rewards : List[float]
            Sequence of rewards received.
        episode_return : float
            Total return for the episode.
        
        Returns
        -------
        Optional[Alert]
            Alert if thresholds exceeded, None otherwise.
        
        Notes
        -----
        During the baseline period (first baseline_episodes), this method
        collects statistics and always returns None.
        """
        self.episode_count += 1
        
        # Update component detectors
        s_value = self._state_detector.add_episode(states)
        if s_value is not None:
            self._latest["S"] = s_value
        
        b_value = self._behavior_detector.add_episode(states, actions, rewards)
        if b_value is not None:
            self._latest["B"] = b_value
        
        p_value = self._performance_monitor.add_episode(episode_return)
        if p_value is not None:
            self._latest["P"] = p_value
        
        # Handle baseline period
        if not self.baseline_complete:
            self._update_baseline()
            if self.episode_count >= self.config.baseline_episodes:
                self._finalize_baseline()
            return None
        
        # Check if we should evaluate
        should_check = self._should_check(s_value, b_value, p_value)
        
        if should_check:
            snapshot = self._create_snapshot()
            self.history.append(snapshot)
            
            alert = self._evaluate_alert(snapshot)
            if alert:
                self.alerts.append(alert)
                return alert
        
        return None
    
    def _should_check(
        self,
        s_value: Optional[float],
        b_value: Optional[float],
        p_value: Optional[float]
    ) -> bool:
        """Determine if we should evaluate for alerts."""
        if self.config.use_sliding_window:
            if self.episode_count - self._last_check_episode >= self.config.stride:
                self._last_check_episode = self.episode_count
                return True
            return False
        else:
            # Fixed windows: check when any component has new value
            return any(v is not None for v in [s_value, b_value, p_value])
    
    def _update_baseline(self) -> None:
        """Collect values during baseline period."""
        for key in ["S", "B", "P"]:
            if self._latest[key] is not None:
                self._baseline_values[key].append(self._latest[key])
    
    def _finalize_baseline(self) -> None:
        """Compute baseline statistics using robust estimators."""
        for key in ["S", "B", "P"]:
            values = self._baseline_values[key]
            if len(values) >= 2:
                # Use median and MAD for robustness to outliers
                median = float(np.median(values))
                mad = float(np.median(np.abs(np.array(values) - median)))
                # Convert MAD to std equivalent (consistency constant for normal)
                std = 1.4826 * mad
                
                # Clamp minimum std to prevent division issues
                if std < self.config.min_std:
                    self._std_clamp_count += 1
                    std = self.config.min_std
                
                self.baseline_stats[key] = (median, std)
            else:
                # Fallback for insufficient data
                self.baseline_stats[key] = (0.0, 1.0)
        
        self.baseline_complete = True
    
    def _normalize(self, component: str, value: float) -> float:
        """Compute z-score using baseline statistics."""
        if component not in self.baseline_stats:
            return 0.0
        mean, std = self.baseline_stats[component]
        return (value - mean) / std
    
    def _create_snapshot(self) -> MonitoringSnapshot:
        """Create monitoring snapshot with current values."""
        z_scores = {}
        for key in ["S", "B", "P"]:
            if self._latest[key] is not None:
                z_scores[key] = self._normalize(key, self._latest[key])
        
        # CDS: mean of absolute z-scores
        valid_z = [abs(z) for z in z_scores.values()]
        composite = float(np.mean(valid_z)) if valid_z else None
        
        alert_level = self._determine_level(z_scores, composite)
        
        return MonitoringSnapshot(
            episode=self.episode_count,
            state_shift=self._latest["S"],
            behavior_shift=self._latest["B"],
            performance_change=self._latest["P"],
            z_scores=z_scores,
            composite_score=composite,
            alert_level=alert_level
        )
    
    def _determine_level(
        self,
        z_scores: Dict[str, float],
        composite: Optional[float]
    ) -> AlertLevel:
        """Determine alert level from scores."""
        if composite is None:
            return AlertLevel.GREEN
        
        # Check composite threshold
        if composite >= self.config.red_threshold:
            return AlertLevel.RED
        if composite >= self.config.yellow_threshold:
            return AlertLevel.YELLOW
        
        # Check individual components
        for z in z_scores.values():
            if abs(z) >= self.config.component_red:
                return AlertLevel.RED
            if abs(z) >= self.config.component_yellow:
                return AlertLevel.YELLOW
        
        return AlertLevel.GREEN
    
    def _evaluate_alert(self, snapshot: MonitoringSnapshot) -> Optional[Alert]:
        """Generate alert if thresholds exceeded."""
        if snapshot.alert_level == AlertLevel.GREEN:
            return None
        
        # Identify triggering components
        triggered_by = []
        for key, z in snapshot.z_scores.items():
            if abs(z) >= self.config.yellow_threshold:
                triggered_by.append(key)
        
        return Alert(
            level=snapshot.alert_level,
            composite_score=snapshot.composite_score or 0.0,
            component_scores=snapshot.z_scores.copy(),
            triggered_by=triggered_by,
            episode=self.episode_count
        )
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - episode: Current episode count
            - baseline_complete: Whether baseline is established
            - baseline_stats: Baseline statistics
            - latest_values: Most recent component values
            - alert_counts: Count of each alert level
            - std_clamp_count: Number of times std was clamped
        """
        return {
            "episode": self.episode_count,
            "baseline_complete": self.baseline_complete,
            "baseline_stats": {
                k: {"mean": v[0], "std": v[1]} 
                for k, v in self.baseline_stats.items()
            },
            "latest_values": self._latest.copy(),
            "alert_counts": {
                "red": sum(1 for a in self.alerts if a.level == AlertLevel.RED),
                "yellow": sum(1 for a in self.alerts if a.level == AlertLevel.YELLOW),
            },
            "std_clamp_count": self._std_clamp_count,
        }
    
    def reset(self) -> None:
        """Reset monitor to initial state.
        
        This clears all history, alerts, and baseline statistics.
        Use this to start fresh monitoring.
        """
        self._state_detector.reset()
        self._behavior_detector.reset()
        self._performance_monitor.reset()
        
        self.baseline_complete = False
        self.baseline_stats = {}
        self._baseline_values = {"S": [], "B": [], "P": []}
        self._latest = {"S": None, "B": None, "P": None}
        
        self.history = []
        self.alerts = []
        self.episode_count = 0
        self._last_check_episode = 0
        self._std_clamp_count = 0
