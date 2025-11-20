"""
langchain-fused-model: Intelligent routing and management for multiple LangChain ChatModel
instances.

This package provides the MultiModelManager class for managing multiple ChatModel instances
with features like intelligent routing, rate limiting, automatic fallback, and structured
output support.
"""

__version__ = "0.1.2"

from .exceptions import (
    AllModelsFailedError,
    MultiModelError,
    RateLimitExceededError,
    StructuredOutputError,
)
from .manager import ModelConfig, MultiModelManager
from .rate_limiter import RateLimiter
from .strategy import RoutingStrategy, StrategySelector
from .structured_output import StructuredOutputHandler
from .usage_tracker import UsageStats, UsageTracker

__all__ = [
    "MultiModelManager",
    "ModelConfig",
    "RoutingStrategy",
    "StrategySelector",
    "UsageTracker",
    "UsageStats",
    "RateLimiter",
    "StructuredOutputHandler",
    "MultiModelError",
    "AllModelsFailedError",
    "RateLimitExceededError",
    "StructuredOutputError",
]
