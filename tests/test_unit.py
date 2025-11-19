"""Unit tests for langchain-fused-model components with mocks."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from langchain_fused_model import (
    AllModelsFailedError,
    ModelConfig,
    MultiModelManager,
    RateLimiter,
    RateLimitExceededError,
    RoutingStrategy,
    StrategySelector,
    StructuredOutputError,
    StructuredOutputHandler,
    UsageStats,
)

# ============================================================================
# RateLimiter Tests
# ============================================================================


class TestRateLimiter:
    """Tests for RateLimiter with mocked time."""

    def test_initialization(self):
        """Test RateLimiter initializes with empty state."""
        limiter = RateLimiter()

        assert limiter._request_times == {}
        assert limiter._cooldowns == {}

    @patch("time.time")
    def test_rpm_enforcement(self, mock_time):
        """Test requests per minute limit enforcement."""
        limiter = RateLimiter()
        config = ModelConfig(max_rpm=3)

        # Start at time 0
        mock_time.return_value = 0.0

        # First 3 requests should be allowed
        for i in range(3):
            assert limiter.is_available(0, config) is True
            limiter.record_request(0)

        # 4th request should be blocked (RPM limit reached)
        assert limiter.is_available(0, config) is False

        # Move time forward by 61 seconds (past the 1-minute window)
        mock_time.return_value = 61.0

        # Should be available again
        assert limiter.is_available(0, config) is True

    @patch("time.time")
    def test_rps_enforcement(self, mock_time):
        """Test requests per second limit enforcement."""
        limiter = RateLimiter()
        config = ModelConfig(max_rps=2)

        # Start at time 0
        mock_time.return_value = 0.0

        # First 2 requests should be allowed
        for i in range(2):
            assert limiter.is_available(0, config) is True
            limiter.record_request(0)

        # 3rd request should be blocked (RPS limit reached)
        assert limiter.is_available(0, config) is False

        # Move time forward by 1.1 seconds
        mock_time.return_value = 1.1

        # Should be available again
        assert limiter.is_available(0, config) is True

    @patch("time.time")
    def test_cooldown_behavior(self, mock_time):
        """Test cooldown period enforcement."""
        limiter = RateLimiter()
        config = ModelConfig()

        # Start at time 0
        mock_time.return_value = 0.0

        # Model should be available initially
        assert limiter.is_available(0, config) is True

        # Set a 30-second cooldown
        limiter.set_cooldown(0, 30.0)

        # Should not be available during cooldown
        assert limiter.is_available(0, config) is False

        # Move time forward by 20 seconds (still in cooldown)
        mock_time.return_value = 20.0
        assert limiter.is_available(0, config) is False

        # Move time forward by 31 seconds (past cooldown)
        mock_time.return_value = 31.0
        assert limiter.is_available(0, config) is True

    @patch("time.time")
    def test_multiple_models_independent_limits(self, mock_time):
        """Test that rate limits are tracked independently per model."""
        limiter = RateLimiter()
        config1 = ModelConfig(max_rpm=2)
        config2 = ModelConfig(max_rpm=2)

        mock_time.return_value = 0.0

        # Use up model 0's limit
        limiter.record_request(0)
        limiter.record_request(0)

        # Model 0 should be at limit
        assert limiter.is_available(0, config1) is False

        # Model 1 should still be available
        assert limiter.is_available(1, config2) is True

    @patch("time.time")
    def test_record_request_stores_timestamp(self, mock_time):
        """Test that record_request stores timestamps correctly."""
        limiter = RateLimiter()

        mock_time.return_value = 100.0
        limiter.record_request(0)

        mock_time.return_value = 200.0
        limiter.record_request(0)

        # Check that timestamps were stored
        assert 0 in limiter._request_times
        assert len(limiter._request_times[0]) == 2
        assert limiter._request_times[0][0] == 100.0
        assert limiter._request_times[0][1] == 200.0

    @patch("time.time")
    def test_old_timestamps_cleaned_up(self, mock_time):
        """Test that old timestamps are removed from tracking."""
        limiter = RateLimiter()
        config = ModelConfig(max_rpm=10)

        # Record requests at time 0
        mock_time.return_value = 0.0
        for i in range(5):
            limiter.record_request(0)

        # Move time forward by 61 seconds
        mock_time.return_value = 61.0

        # Check availability (this should clean up old timestamps)
        limiter.is_available(0, config)

        # Old timestamps should be removed
        assert len(limiter._request_times[0]) == 0


# ============================================================================
# RoutingStrategy Tests
# ============================================================================


class TestRoutingStrategies:
    """Tests for routing strategy logic."""

    def test_priority_strategy(self):
        """Test PRIORITY strategy selects highest priority model."""
        selector = StrategySelector()

        configs = [
            ModelConfig(priority=1),
            ModelConfig(priority=5),  # Highest
            ModelConfig(priority=3),
        ]

        available_models = [0, 1, 2]
        usage_stats = {}

        selected = selector.select(
            RoutingStrategy.PRIORITY, [], configs, usage_stats, available_models
        )

        assert selected == 1  # Model with priority 5

    def test_priority_strategy_with_limited_availability(self):
        """Test PRIORITY strategy with only some models available."""
        selector = StrategySelector()

        configs = [
            ModelConfig(priority=10),  # Highest but not available
            ModelConfig(priority=5),
            ModelConfig(priority=3),
        ]

        # Only models 1 and 2 are available
        available_models = [1, 2]
        usage_stats = {}

        selected = selector.select(
            RoutingStrategy.PRIORITY, [], configs, usage_stats, available_models
        )

        assert selected == 1  # Highest priority among available

    def test_round_robin_strategy(self):
        """Test ROUND_ROBIN strategy distributes evenly."""
        selector = StrategySelector()

        configs = [ModelConfig(), ModelConfig(), ModelConfig()]
        available_models = [0, 1, 2]
        usage_stats = {}

        # Should cycle through models
        selected1 = selector.select(
            RoutingStrategy.ROUND_ROBIN, [], configs, usage_stats, available_models
        )
        selected2 = selector.select(
            RoutingStrategy.ROUND_ROBIN, [], configs, usage_stats, available_models
        )
        selected3 = selector.select(
            RoutingStrategy.ROUND_ROBIN, [], configs, usage_stats, available_models
        )
        selected4 = selector.select(
            RoutingStrategy.ROUND_ROBIN, [], configs, usage_stats, available_models
        )

        assert selected1 == 0
        assert selected2 == 1
        assert selected3 == 2
        assert selected4 == 0  # Wraps around

    def test_least_used_strategy(self):
        """Test LEAST_USED strategy selects model with lowest usage."""
        selector = StrategySelector()

        configs = [ModelConfig(), ModelConfig(), ModelConfig()]
        available_models = [0, 1, 2]

        # Set up usage stats
        usage_stats = {
            0: UsageStats(total_requests=10),
            1: UsageStats(total_requests=3),  # Least used
            2: UsageStats(total_requests=7),
        }

        selected = selector.select(
            RoutingStrategy.LEAST_USED, [], configs, usage_stats, available_models
        )

        assert selected == 1  # Model with 3 requests

    def test_least_used_strategy_with_no_stats(self):
        """Test LEAST_USED strategy when some models have no stats."""
        selector = StrategySelector()

        configs = [ModelConfig(), ModelConfig(), ModelConfig()]
        available_models = [0, 1, 2]

        # Only model 0 has stats
        usage_stats = {
            0: UsageStats(total_requests=10),
        }

        selected = selector.select(
            RoutingStrategy.LEAST_USED, [], configs, usage_stats, available_models
        )

        # Should select model 1 or 2 (both have 0 requests)
        assert selected in [1, 2]

    def test_cost_aware_strategy(self):
        """Test COST_AWARE strategy selects lowest cost model."""
        selector = StrategySelector()

        configs = [
            ModelConfig(cost_per_1k_tokens=0.03),
            ModelConfig(cost_per_1k_tokens=0.01),  # Cheapest
            ModelConfig(cost_per_1k_tokens=0.02),
        ]

        available_models = [0, 1, 2]
        usage_stats = {}

        selected = selector.select(
            RoutingStrategy.COST_AWARE, [], configs, usage_stats, available_models
        )

        assert selected == 1  # Model with cost 0.01

    def test_custom_callable_strategy(self):
        """Test custom callable strategy."""
        selector = StrategySelector()

        # Custom strategy that always selects the last model
        def custom_strategy(models, configs, usage_stats, available_models):
            return available_models[-1]

        configs = [ModelConfig(), ModelConfig(), ModelConfig()]
        available_models = [0, 1, 2]
        usage_stats = {}

        selected = selector.select(custom_strategy, [], configs, usage_stats, available_models)

        assert selected == 2

    def test_strategy_with_empty_available_models_raises_error(self):
        """Test that selecting with no available models raises ValueError."""
        selector = StrategySelector()

        configs = [ModelConfig()]
        available_models = []
        usage_stats = {}

        with pytest.raises(ValueError, match="No available models"):
            selector.select(RoutingStrategy.PRIORITY, [], configs, usage_stats, available_models)


# ============================================================================
# StructuredOutput Tests
# ============================================================================


class TestStructuredOutput:
    """Tests for structured output handling."""

    def test_has_native_support_detection(self):
        """Test detection of native structured output support."""
        handler = StructuredOutputHandler()

        # Mock model with native support
        model_with_support = Mock()
        model_with_support.with_structured_output = Mock()

        assert handler.has_native_support(model_with_support) is True

        # Mock model without native support
        model_without_support = Mock(spec=[])

        assert handler.has_native_support(model_without_support) is False

    def test_json_extraction_from_clean_response(self):
        """Test JSON extraction from clean JSON response."""
        handler = StructuredOutputHandler()

        response = '{"name": "John", "age": 30}'
        extracted = handler._extract_json(response)

        assert extracted == '{"name": "John", "age": 30}'

    def test_json_extraction_with_surrounding_text(self):
        """Test JSON extraction when JSON is embedded in text."""
        handler = StructuredOutputHandler()

        response = 'Here is the data: {"name": "John", "age": 30} Hope this helps!'
        extracted = handler._extract_json(response)

        assert extracted == '{"name": "John", "age": 30}'

    def test_json_extraction_with_nested_objects(self):
        """Test JSON extraction with nested objects."""
        handler = StructuredOutputHandler()

        response = '{"person": {"name": "John", "age": 30}, "city": "NYC"}'
        extracted = handler._extract_json(response)

        assert extracted == '{"person": {"name": "John", "age": 30}, "city": "NYC"}'

    def test_json_extraction_array(self):
        """Test JSON extraction for arrays."""
        handler = StructuredOutputHandler()

        # Test with simple array at start
        response = '[{"name": "John"}, {"name": "Jane"}]'
        extracted = handler._extract_json(response)

        # Should extract valid JSON (either the array or an object from it)
        assert extracted.startswith("{") or extracted.startswith("[")

    def test_json_extraction_fails_with_no_json(self):
        """Test that extraction fails when no JSON is present."""
        handler = StructuredOutputHandler()

        response = "This is just plain text with no JSON"

        with pytest.raises(StructuredOutputError, match="Could not extract valid JSON"):
            handler._extract_json(response)

    def test_pydantic_validation_success(self):
        """Test successful Pydantic validation."""
        handler = StructuredOutputHandler()

        class Person(BaseModel):
            name: str
            age: int

        json_str = '{"name": "John", "age": 30}'
        result = handler._parse_to_model(json_str, Person)

        assert isinstance(result, Person)
        assert result.name == "John"
        assert result.age == 30

    def test_pydantic_validation_failure(self):
        """Test Pydantic validation failure."""
        handler = StructuredOutputHandler()

        class Person(BaseModel):
            name: str
            age: int

        # Missing required field
        json_str = '{"name": "John"}'

        with pytest.raises(StructuredOutputError, match="Failed to validate"):
            handler._parse_to_model(json_str, Person)

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        handler = StructuredOutputHandler()

        class Person(BaseModel):
            name: str

        invalid_json = '{"name": "John"'  # Missing closing brace

        with pytest.raises(StructuredOutputError, match="Failed to decode JSON"):
            handler._parse_to_model(invalid_json, Person)

    def test_inject_format_instructions(self):
        """Test that format instructions are injected into messages."""
        handler = StructuredOutputHandler()

        class Person(BaseModel):
            name: str
            age: int

        original_messages = [HumanMessage(content="Tell me about John")]

        modified_messages = handler._inject_format_instructions(original_messages, Person)

        # Should have one more message (system message with instructions)
        assert len(modified_messages) == len(original_messages) + 1

        # First message should be system message with instructions
        assert modified_messages[0].type == "system"
        assert "JSON" in modified_messages[0].content
        assert "schema" in modified_messages[0].content.lower()


# ============================================================================
# MultiModelManager Tests
# ============================================================================


class TestMultiModelManager:
    """Tests for MultiModelManager with mocked dependencies."""

    def test_initialization_with_single_model(self, dummy_chat_model):
        """Test manager initializes with a single model."""
        manager = MultiModelManager(models=[dummy_chat_model])

        assert len(manager.models) == 1
        assert len(manager.model_configs) == 1
        assert manager.strategy == "priority"
        assert manager.default_fallback is True

    def test_initialization_with_multiple_models(self, multiple_dummy_models):
        """Test manager initializes with multiple models."""
        configs = [
            ModelConfig(priority=1),
            ModelConfig(priority=2),
            ModelConfig(priority=3),
        ]

        manager = MultiModelManager(
            models=multiple_dummy_models, model_configs=configs, strategy="round_robin"
        )

        assert len(manager.models) == 3
        assert len(manager.model_configs) == 3
        assert manager.strategy == "round_robin"

    def test_initialization_fails_with_empty_models(self):
        """Test that initialization fails with empty models list."""
        with pytest.raises(ValueError, match="At least one model"):
            MultiModelManager(models=[])

    def test_initialization_fails_with_mismatched_configs(self, multiple_dummy_models):
        """Test that initialization fails when configs don't match models."""
        configs = [ModelConfig()]  # Only one config for three models

        with pytest.raises(ValueError, match="must match"):
            MultiModelManager(models=multiple_dummy_models, model_configs=configs)

    def test_llm_type_property(self, dummy_chat_model):
        """Test _llm_type property returns correct identifier."""
        manager = MultiModelManager(models=[dummy_chat_model])

        assert manager._llm_type == "multi-model-manager"

    def test_identifying_params_property(self, multiple_dummy_models):
        """Test _identifying_params returns correct information."""
        manager = MultiModelManager(models=multiple_dummy_models, strategy="priority")

        params = manager._identifying_params

        assert params["manager_type"] == "multi-model-manager"
        assert params["num_models"] == 3
        assert params["strategy"] == "priority"
        assert len(params["model_types"]) == 3

    def test_select_model_uses_strategy(self, multiple_dummy_models):
        """Test that _select_model uses the configured strategy."""
        configs = [
            ModelConfig(priority=1),
            ModelConfig(priority=3),  # Highest priority
            ModelConfig(priority=2),
        ]

        manager = MultiModelManager(
            models=multiple_dummy_models, model_configs=configs, strategy="priority"
        )

        # Call _select_model
        selected_model, selected_idx = manager._select_model()

        # Should select model with highest priority (index 1)
        assert selected_idx == 1
        assert selected_model == multiple_dummy_models[1]

    def test_select_model_raises_when_all_rate_limited(self, multiple_dummy_models):
        """Test that _select_model raises error when all models are rate limited."""
        # Create configs with very low rate limits
        configs = [
            ModelConfig(max_rps=1),
            ModelConfig(max_rps=1),
            ModelConfig(max_rps=1),
        ]

        manager = MultiModelManager(models=multiple_dummy_models, model_configs=configs)

        # Exhaust all rate limits
        for i in range(3):
            manager._rate_limiter.record_request(i)

        # Should raise RateLimitExceededError
        with pytest.raises(RateLimitExceededError, match="rate limited"):
            manager._select_model()

    def test_invoke_with_fallback_success(self, dummy_chat_model):
        """Test successful invocation without fallback."""
        manager = MultiModelManager(models=[dummy_chat_model])

        # Call _invoke_with_fallback
        messages = [HumanMessage(content="Test")]
        result = manager._invoke_with_fallback(0, messages)

        # Verify success
        assert result is not None
        assert len(result.generations) > 0
        assert result.generations[0].message.content == "This is a dummy response"

    def test_invoke_with_fallback_on_error(self, rate_limited_model, dummy_chat_model):
        """Test fallback to next model on error."""
        manager = MultiModelManager(models=[rate_limited_model, dummy_chat_model])

        # Call _invoke_with_fallback starting with failing model
        messages = [HumanMessage(content="Test")]
        result = manager._invoke_with_fallback(0, messages)

        # Verify fallback occurred and succeeded
        assert result is not None
        assert result.generations[0].message.content == "This is a dummy response"

    def test_invoke_with_fallback_all_fail(self, rate_limited_model, timeout_model):
        """Test that AllModelsFailedError is raised when all models fail."""
        manager = MultiModelManager(models=[rate_limited_model, timeout_model])

        # Should raise AllModelsFailedError
        messages = [HumanMessage(content="Test")]
        with pytest.raises(AllModelsFailedError):
            manager._invoke_with_fallback(0, messages)

    def test_fallback_disabled(self, rate_limited_model):
        """Test that fallback doesn't occur when disabled."""
        manager = MultiModelManager(models=[rate_limited_model], default_fallback=False)

        # Should raise the original exception (not AllModelsFailedError)
        messages = [HumanMessage(content="Test")]
        with pytest.raises(Exception, match="Rate limit"):
            manager._invoke_with_fallback(0, messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
