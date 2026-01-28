"""
SMF-AAS: Strategy Monitoring Framework for Adaptive AI Systems

A domain-agnostic framework for post-deployment monitoring of AI systems
that can modify their strategies, designed for EU AI Act Article 72 compliance.

References
----------
.. [1] EU AI Act, Article 72: Post-market monitoring by providers
.. [2] Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
.. [3] Lin, J. (1991). Divergence measures based on the Shannon entropy.

Example
-------
>>> from smf_aas import StrategyMonitor, MonitorConfig
>>> config = MonitorConfig(window_size=50, baseline_episodes=200)
>>> monitor = StrategyMonitor(config, n_actions=9)
>>> alert = monitor.update(states, actions, rewards, episode_return)
"""

__version__ = "1.0.0"
__author__ = "B. Kleibrink"
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
