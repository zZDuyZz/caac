"""Scheduler flags -- overhead optimisations, togglable independently of decision logic."""
from caac.scheduler.lazy import lazy_skip
from caac.scheduler.spacing import AdaptiveSpacing
__all__ = ["AdaptiveSpacing", "lazy_skip"]
