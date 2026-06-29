"""RL package for automation and training.

The game package must not import this package. Automation code should depend on
stable interfaces exposed by daxigua, never the other way around.
"""

from .env import DaxiguaEnv, DaxiguaEnvConfig


__all__ = ['DaxiguaEnv', 'DaxiguaEnvConfig']
