"""
Metrics module for SMF-AAS monitoring components.

This module provides the three core detection components:

- :class:`StateDistributionShift`: Detects shifts in state visitation patterns
- :class:`BehaviorShiftDetector`: Detects changes in behavioral features
- :class:`PerformanceMonitor`: Detects changes in episode performance

Each component operates on sliding or fixed windows and outputs a scalar
metric that can be z-score normalized against a baseline.
"""

from .state_shift import StateDistributionShift
from .behavior_shift import BehaviorShiftDetector
from .performance import PerformanceMonitor

__all__ = [
    "StateDistributionShift",
    "BehaviorShiftDetector",
    "PerformanceMonitor",
]
