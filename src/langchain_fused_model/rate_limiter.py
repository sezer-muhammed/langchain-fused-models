"""Rate limiting functionality for MultiModelManager."""

import logging
import time
from collections import deque
from typing import Dict, Optional

from .manager import ModelConfig

# Set up logger
logger = logging.getLogger(__name__)


class RateLimiter:
    """Tracks and enforces rate limits per model."""

    def __init__(self):
        """Initialize rate limiter with empty tracking dictionaries."""
        self._request_times: Dict[int, deque] = {}
        self._cooldowns: Dict[int, float] = {}

    def is_available(self, model_idx: int, config: ModelConfig) -> bool:
        """
        Check if model is available (not rate limited or in cooldown).

        Args:
            model_idx: Index of the model to check
            config: Configuration for the model including rate limits

        Returns:
            True if model is available, False otherwise
        """
        current_time = time.time()

        # Check if model is in cooldown
        if model_idx in self._cooldowns:
            if current_time < self._cooldowns[model_idx]:
                remaining = self._cooldowns[model_idx] - current_time
                logger.info(
                    f"Model {model_idx} is in cooldown for {remaining:.1f} more seconds"
                )
                return False
            else:
                # Cooldown expired, remove it
                logger.info(f"Model {model_idx} cooldown expired, now available")
                del self._cooldowns[model_idx]

        # Check RPM limit if configured
        if config.max_rpm is not None:
            if not self._check_rpm(model_idx, config.max_rpm):
                logger.info(
                    f"Model {model_idx} has reached RPM limit of {config.max_rpm}"
                )
                return False

        # Check RPS limit if configured
        if config.max_rps is not None:
            if not self._check_rps(model_idx, config.max_rps):
                logger.info(
                    f"Model {model_idx} has reached RPS limit of {config.max_rps}"
                )
                return False

        return True

    def record_request(self, model_idx: int) -> None:
        """
        Record a request timestamp for rate tracking.

        Args:
            model_idx: Index of the model that received the request
        """
        current_time = time.time()

        if model_idx not in self._request_times:
            self._request_times[model_idx] = deque()

        self._request_times[model_idx].append(current_time)

    def set_cooldown(self, model_idx: int, duration: float) -> None:
        """
        Set temporary cooldown period for a model.

        Args:
            model_idx: Index of the model to set cooldown for
            duration: Cooldown duration in seconds
        """
        self._cooldowns[model_idx] = time.time() + duration
        logger.info(f"Model {model_idx} cooldown set for {duration} seconds")

    def _check_rpm(self, model_idx: int, max_rpm: int) -> bool:
        """
        Check if model is within requests per minute limit.

        Args:
            model_idx: Index of the model to check
            max_rpm: Maximum requests per minute allowed

        Returns:
            True if within limit, False otherwise
        """
        if model_idx not in self._request_times:
            return True

        current_time = time.time()
        one_minute_ago = current_time - 60.0

        # Remove timestamps older than 1 minute
        request_times = self._request_times[model_idx]
        while request_times and request_times[0] < one_minute_ago:
            request_times.popleft()

        # Check if we're at or over the limit
        return len(request_times) < max_rpm

    def _check_rps(self, model_idx: int, max_rps: int) -> bool:
        """
        Check if model is within requests per second limit.

        Args:
            model_idx: Index of the model to check
            max_rps: Maximum requests per second allowed

        Returns:
            True if within limit, False otherwise
        """
        if model_idx not in self._request_times:
            return True

        current_time = time.time()
        one_second_ago = current_time - 1.0

        # Remove timestamps older than 1 second
        request_times = self._request_times[model_idx]
        while request_times and request_times[0] < one_second_ago:
            request_times.popleft()

        # Check if we're at or over the limit
        return len(request_times) < max_rps
