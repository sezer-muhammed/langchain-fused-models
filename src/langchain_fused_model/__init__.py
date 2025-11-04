"""
langchain-fused-model: Intelligent routing and management for multiple LangChain ChatModel instances.

This package provides the MultiModelManager class for managing multiple ChatModel instances
with features like intelligent routing, rate limiting, automatic fallback, and structured output support.
"""

__version__ = "0.1.0"

from .manager import MultiModelManager, ModelConfig
from .strategy import RoutingStrategy, StrategySelector
from .usage_tracker import UsageTracker, UsageStats
from .rate_limiter import RateLimiter
from .structured_output import StructuredOutputHandler
from .exceptions import (
    MultiModelError,
    AllModelsFailedError,
    RateLimitExceededError,
    StructuredOutputError,
)

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
