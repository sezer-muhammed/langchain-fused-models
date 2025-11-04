"""Routing strategies for model selection."""

from enum import Enum
from typing import List, Dict, Callable, Optional, Any
from .usage_tracker import UsageStats
from .manager import ModelConfig


class RoutingStrategy(Enum):
    """Enumeration of available routing strategies.
    
    Attributes:
        PRIORITY: Select models by priority order (highest first).
        ROUND_ROBIN: Distribute requests evenly across models.
        LEAST_USED: Select the least-utilized model.
        COST_AWARE: Prioritize lower-cost models.
    """
    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"
    LEAST_USED = "least_used"
    COST_AWARE = "cost_aware"


class StrategySelector:
    """Selects models based on configured routing strategy.
    
    This class implements various routing strategies for distributing
    requests across multiple ChatModel instances. It supports both
    built-in strategies and custom callable strategies.
    """
    
    def __init__(self):
        """Initialize the strategy selector."""
        self._round_robin_counter: int = 0
    
    def select(
        self,
        strategy: RoutingStrategy | Callable,
        models: List[Any],
        configs: List[ModelConfig],
        usage_stats: Dict[int, UsageStats],
        available_models: List[int]
    ) -> int:
        """Select a model based on the configured strategy.
        
        Args:
            strategy: The routing strategy to use (enum or callable).
            models: List of all ChatModel instances.
            configs: List of ModelConfig objects for each model.
            usage_stats: Dictionary mapping model indices to UsageStats.
            available_models: List of model indices that are currently available.
        
        Returns:
            Index of the selected model.
        
        Raises:
            ValueError: If no models are available or strategy is invalid.
        """
        if not available_models:
            raise ValueError("No available models to select from")
        
        # Handle custom callable strategy
        if callable(strategy):
            return strategy(models, configs, usage_stats, available_models)
        
        # Convert string to enum if needed
        if isinstance(strategy, str):
            try:
                strategy = RoutingStrategy(strategy)
            except ValueError:
                raise ValueError(f"Unknown routing strategy: {strategy}")
        
        # Handle built-in strategies
        if strategy == RoutingStrategy.PRIORITY:
            return self._priority_select(configs, available_models)
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_models)
        elif strategy == RoutingStrategy.LEAST_USED:
            return self._least_used_select(usage_stats, available_models)
        elif strategy == RoutingStrategy.COST_AWARE:
            return self._cost_aware_select(configs, available_models)
        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")

    def _priority_select(
        self,
        configs: List[ModelConfig],
        available_models: List[int]
    ) -> int:
        """Select the highest priority available model.
        
        Args:
            configs: List of ModelConfig objects for each model.
            available_models: List of model indices that are currently available.
        
        Returns:
            Index of the model with the highest priority.
        """
        # Find the model with the highest priority among available models
        best_model = available_models[0]
        best_priority = configs[best_model].priority
        
        for model_idx in available_models[1:]:
            priority = configs[model_idx].priority
            if priority > best_priority:
                best_priority = priority
                best_model = model_idx
        
        return best_model
    
    def _round_robin_select(
        self,
        available_models: List[int]
    ) -> int:
        """Select the next model in rotation.
        
        Distributes requests evenly across available models using
        a round-robin approach.
        
        Args:
            available_models: List of model indices that are currently available.
        
        Returns:
            Index of the next model in the rotation.
        """
        # Select model based on counter and wrap around
        selected_idx = self._round_robin_counter % len(available_models)
        self._round_robin_counter += 1
        
        return available_models[selected_idx]
    
    def _least_used_select(
        self,
        usage_stats: Dict[int, UsageStats],
        available_models: List[int]
    ) -> int:
        """Select the model with the lowest usage count.
        
        Args:
            usage_stats: Dictionary mapping model indices to UsageStats.
            available_models: List of model indices that are currently available.
        
        Returns:
            Index of the least-utilized model.
        """
        # Find the model with the lowest total requests
        best_model = available_models[0]
        best_usage = usage_stats.get(best_model, UsageStats()).total_requests
        
        for model_idx in available_models[1:]:
            usage = usage_stats.get(model_idx, UsageStats()).total_requests
            if usage < best_usage:
                best_usage = usage
                best_model = model_idx
        
        return best_model
    
    def _cost_aware_select(
        self,
        configs: List[ModelConfig],
        available_models: List[int]
    ) -> int:
        """Select the lowest cost available model.
        
        Prioritizes models with lower cost_per_1k_tokens values.
        
        Args:
            configs: List of ModelConfig objects for each model.
            available_models: List of model indices that are currently available.
        
        Returns:
            Index of the model with the lowest cost.
        """
        # Find the model with the lowest cost among available models
        best_model = available_models[0]
        best_cost = configs[best_model].cost_per_1k_tokens
        
        for model_idx in available_models[1:]:
            cost = configs[model_idx].cost_per_1k_tokens
            if cost < best_cost:
                best_cost = cost
                best_model = model_idx
        
        return best_model
