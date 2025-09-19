"""
Options Framework Implementation
===============================

Temporal abstractions through options and semi-MDPs.
Based on "Between MDPs and Semi-MDPs" (Sutton et al., 1999).
"""

from .options_agent import OptionsAgent
from .options_config import OptionsConfig

__all__ = ["OptionsAgent", "OptionsConfig"]