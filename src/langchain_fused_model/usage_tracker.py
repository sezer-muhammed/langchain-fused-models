"""Usage tracking for managed ChatModel instances."""

from dataclasses import dataclass, field
from typing import Dict, Optional
import time


@dataclass
class UsageStats:
    """Statistics for a managed ChatModel instance.
    
    Attributes:
        total_requests: Total number of requests made to this model.
        successful_requests: Number of successful requests.
        failed_requests: Number of failed requests.
        total_tokens: Total tokens used by this model.
        last_used: Unix timestamp of last request, or None if never used.
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    last_used: Optional[float] = None


class UsageTracker:
    """Tracks usage statistics for each managed ChatModel.
    
    This class maintains statistics for each model including request counts,
    success/failure rates, token usage, and last usage timestamp.
    """
    
    def __init__(self):
        """Initialize the usage tracker with an empty statistics dictionary."""
        self._stats: Dict[int, UsageStats] = {}
    
    def record_request(
        self,
        model_idx: int,
        success: bool,
        tokens: int = 0
    ) -> None:
        """Record a request outcome and update statistics.
        
        Args:
            model_idx: Index of the model in the manager's model list.
            success: Whether the request was successful.
            tokens: Number of tokens used in the request. Default is 0.
        """
        # Initialize stats for this model if not exists
        if model_idx not in self._stats:
            self._stats[model_idx] = UsageStats()
        
        stats = self._stats[model_idx]
        
        # Update statistics
        stats.total_requests += 1
        if success:
            stats.successful_requests += 1
        else:
            stats.failed_requests += 1
        stats.total_tokens += tokens
        stats.last_used = time.time()
    
    def get_stats(self, model_idx: int) -> UsageStats:
        """Get usage statistics for a specific model.
        
        Args:
            model_idx: Index of the model in the manager's model list.
        
        Returns:
            UsageStats object for the specified model. Returns a new UsageStats
            instance with default values if the model has no recorded usage.
        """
        return self._stats.get(model_idx, UsageStats())
    
    def get_all_stats(self) -> Dict[int, UsageStats]:
        """Get usage statistics for all models.
        
        Returns:
            Dictionary mapping model indices to their UsageStats objects.
        """
        return self._stats.copy()
