"""
SMF-AAS: Strategy Monitoring Framework for Adaptive AI Systems

A domain-agnostic framework for post-deployment monitoring of AI systems
that can modify their strategies, designed for EU AI Act Article 72 compliance.

"""

__version__ = ""
__author__ = ""
__email__ = ""

from .monitor import StrategyMonitor, MonitorConfig, Alert, AlertLevel
from .environments import get_env, list_envs, GameEnvironment, GameState

__all__ = [
    # Core monitoring
    "StrategyMonitor",
    "MonitorConfig", 
    "Alert",
    "AlertLevel",
    # Environments
    "get_env",
    "list_envs",
    "GameEnvironment",
    "GameState",
    # Metadata
    "__version__",
]
