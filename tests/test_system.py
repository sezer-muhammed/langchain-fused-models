"""System tests for langchain-fused-model with MockChatModel integration.

These tests verify the complete system behavior with real LangChain components,
testing all routing strategies, rate limiting, fallback mechanisms, structured
output, and LangChain integration features.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from langchain_fused_model import (
    AllModelsFailedError,
    ModelConfig,
    MultiModelManager,
    RateLimitExceededError,
)

# ============================================================================
# System Tests: Routing Strategies
# ============================================================================


class TestRoutingStrategiesSystem:
    """System tests for all routing strategies with real models."""

    def test_priority_routing_with_multiple_models(self, multiple_dummy_models):
        """Test PRIORITY strategy selects highest priority model."""
        configs = [
            ModelConfig(priority=1),
            ModelConfig(priority=5),  # Highest priority
            ModelConfig(priority=3),
        ]

        manager = MultiModelManager(
            models=multiple_dummy_models, model_configs=configs, strategy="priority"
        )

        # Invoke multiple times - should always use model 1 (highest priority)
        for _ in range(3):
            result = manager.invoke("Test message")
            assert "Response from model 2" in result.content

    def test_round_robin_routing_distributes_evenly(self, multiple_dummy_models):
        """Test ROUND_ROBIN strategy distributes requests evenly."""
        manager = MultiModelManager(models=multiple_dummy_models, strategy="round_robin")

        # Make multiple requests
        results = []
        for _ in range(6):
            result = manager.invoke("Test message")
            results.append(result.content)

        # Should cycle through models: 1, 2, 3, 1, 2, 3
        assert results[0] == "Response from model 1"
        assert results[1] == "Response from model 2"
        assert results[2] == "Response from model 3"
        assert results[3] == "Response from model 1"
        assert results[4] == "Response from model 2"
        assert results[5] == "Response from model 3"

    def test_least_used_routing_balances_load(self, multiple_dummy_models):
        """Test LEAST_USED strategy balances load across models."""
        manager = MultiModelManager(models=multiple_dummy_models, strategy="least_used")

        # Make multiple requests
        results = []
        for _ in range(6):
            result = manager.invoke("Test message")
            results.append(result.content)

        # Each model should be used roughly equally
        # Count occurrences of each model's response
        model1_count = sum(1 for r in results if "model 1" in r)
        model2_count = sum(1 for r in results if "model 2" in r)
        model3_count = sum(1 for r in results if "model 3" in r)

        # Each should be used exactly twice (6 requests / 3 models)
        assert model1_count == 2
        assert model2_count == 2
        assert model3_count == 2

    def test_cost_aware_routing_prefers_cheaper_models(self, multiple_dummy_models):
        """Test COST_AWARE strategy prefers lower cost models."""
        configs = [
            ModelConfig(cost_per_1k_tokens=0.03),  # Most expensive
            ModelConfig(cost_per_1k_tokens=0.01),  # Cheapest
            ModelConfig(cost_per_1k_tokens=0.02),  # Middle
        ]

        manager = MultiModelManager(
            models=multiple_dummy_models, model_configs=configs, strategy="cost_aware"
        )

        # Make multiple requests - should always use model 1 (cheapest)
        for _ in range(3):
            result = manager.invoke("Test message")
            assert "Response from model 2" in result.content


# ============================================================================
# System Tests: Rate Limiting
# ============================================================================


class TestRateLimitingSystem:
    """System tests for rate limiting with real models."""

    def test_rate_limiting_triggers_fallback(self, multiple_dummy_models):
        """Test that rate limiting triggers automatic fallback."""
        configs = [
            ModelConfig(priority=10, max_rps=2),  # Highest priority but limited
            ModelConfig(priority=5),  # Fallback model
            ModelConfig(priority=1),
        ]

        manager = MultiModelManager(
            models=multiple_dummy_models, model_configs=configs, strategy="priority"
        )

        # First 2 requests should use model 0 (highest priority)
        result1 = manager.invoke("Test 1")
        result2 = manager.invoke("Test 2")

        assert "Response from model 1" in result1.content
        assert "Response from model 1" in result2.content

        # 3rd request should fallback to model 1 (rate limit hit)
        result3 = manager.invoke("Test 3")
        assert "Response from model 2" in result3.content

    def test_rpm_limit_enforcement(self, multiple_dummy_models):
        """Test requests per minute limit enforcement."""
        configs = [
            ModelConfig(priority=10, max_rpm=3),
            ModelConfig(priority=5),
        ]

        manager = MultiModelManager(
            models=multiple_dummy_models[:2], model_configs=configs, strategy="priority"
        )

        # Make 3 requests quickly (within RPM limit)
        for i in range(3):
            result = manager.invoke(f"Test {i}")
            assert "Response from model 1" in result.content

        # 4th request should fallback due to RPM limit
        result = manager.invoke("Test 4")
        assert "Response from model 2" in result.content

    def test_all_models_rate_limited_raises_error(self, multiple_dummy_models):
        """Test that error is raised when all models are rate limited."""
        configs = [
            ModelConfig(max_rps=1),
            ModelConfig(max_rps=1),
            ModelConfig(max_rps=1),
        ]

        manager = MultiModelManager(
            models=multiple_dummy_models, model_configs=configs, strategy="round_robin"
        )

        # Use up all rate limits
        manager.invoke("Test 1")
        manager.invoke("Test 2")
        manager.invoke("Test 3")

        # Next request should fail with RateLimitExceededError
        with pytest.raises(RateLimitExceededError):
            manager.invoke("Test 4")


# ============================================================================
# System Tests: Error Fallback
# ============================================================================


class TestErrorFallbackSystem:
    """System tests for error fallback mechanisms."""

    def test_fallback_on_model_failure(self, rate_limited_model, dummy_chat_model):
        """Test automatic fallback when primary model fails."""
        manager = MultiModelManager(models=[rate_limited_model, dummy_chat_model])

        # Should fallback to second model when first fails
        result = manager.invoke("Test message")
        assert result.content == "This is a dummy response"

    def test_fallback_through_multiple_failures(
        self, rate_limited_model, timeout_model, dummy_chat_model
    ):
        """Test fallback through multiple failing models."""
        # Use round-robin to ensure we try each model in sequence
        manager = MultiModelManager(
            models=[rate_limited_model, timeout_model, dummy_chat_model], strategy="round_robin"
        )

        # First request: tries model 0 (fails), falls back to model 1 (fails),
        # then model 2 (succeeds)
        result = manager.invoke("Test message")
        assert result.content == "This is a dummy response"

    def test_all_models_failed_error(self, rate_limited_model, timeout_model):
        """Test AllModelsFailedError when all models fail."""
        manager = MultiModelManager(models=[rate_limited_model, timeout_model])

        # Should raise AllModelsFailedError with details
        with pytest.raises(AllModelsFailedError) as exc_info:
            manager.invoke("Test message")

        # Verify error contains information about all failures
        error = exc_info.value
        assert hasattr(error, "errors")
        assert len(error.errors) == 2

    def test_fallback_disabled(self, rate_limited_model, dummy_chat_model):
        """Test that fallback doesn't occur when disabled."""
        manager = MultiModelManager(
            models=[rate_limited_model, dummy_chat_model], default_fallback=False
        )

        # Should raise the original exception, not fallback
        with pytest.raises(Exception, match="Rate limit"):
            manager.invoke("Test message")


# ============================================================================
# System Tests: Structured Output
# ============================================================================


class TestStructuredOutputSystem:
    """System tests for structured output with real models."""

    def test_structured_output_with_native_support(self, structured_output_model):
        """Test structured output with model that has native support."""

        class Person(BaseModel):
            name: str
            age: int

        manager = MultiModelManager(models=[structured_output_model])

        # Create structured output chain
        structured_chain = manager.with_structured_output(Person)

        # Invoke with message
        result = structured_chain.invoke("Tell me about John")

        # Verify result is a Pydantic model instance
        assert isinstance(result, Person)
        assert result.name == "John"
        assert result.age == 30

    def test_structured_output_with_fallback_parsing(self, dummy_chat_model):
        """Test structured output with fallback JSON parsing."""

        class Person(BaseModel):
            name: str
            age: int

        # Create a model that returns JSON in its response
        from tests.conftest import DummyChatModel

        json_model = DummyChatModel(
            model_name="json-model", response_text='{"name": "Alice", "age": 25}'
        )

        manager = MultiModelManager(models=[json_model])

        # Create structured output chain
        structured_chain = manager.with_structured_output(Person)

        # Invoke with message
        result = structured_chain.invoke("Tell me about Alice")

        # Verify result is parsed correctly
        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 25

    def test_structured_output_with_embedded_json(self):
        """Test structured output when JSON is embedded in text."""

        class Person(BaseModel):
            name: str
            age: int

        from tests.conftest import DummyChatModel

        embedded_json_model = DummyChatModel(
            model_name="embedded",
            response_text='Here is the information: {"name": "Bob", "age": 35} Hope this helps!',
        )

        manager = MultiModelManager(models=[embedded_json_model])
        structured_chain = manager.with_structured_output(Person)

        result = structured_chain.invoke("Tell me about Bob")

        assert isinstance(result, Person)
        assert result.name == "Bob"
        assert result.age == 35

    def test_structured_output_fallback_on_native_failure(self):
        """Test fallback to parsing when native structured output fails."""

        class Person(BaseModel):
            name: str
            age: int

        # Create a model with native support that returns valid JSON
        from tests.conftest import StructuredOutputChatModel

        model = StructuredOutputChatModel(model_name="test")

        manager = MultiModelManager(models=[model])
        structured_chain = manager.with_structured_output(Person)

        result = structured_chain.invoke("Tell me about someone")

        assert isinstance(result, Person)
        assert result.name == "John"
        assert result.age == 30


# ============================================================================
# System Tests: LangChain Integration
# ============================================================================


class TestLangChainIntegrationSystem:
    """System tests for LangChain Runnable interface integration."""

    def test_invoke_method(self, dummy_chat_model):
        """Test invoke method works correctly."""
        manager = MultiModelManager(models=[dummy_chat_model])

        # Test with string input
        result = manager.invoke("Hello, world!")
        assert isinstance(result, AIMessage)
        assert result.content == "This is a dummy response"

        # Test with message list input
        messages = [HumanMessage(content="Test")]
        result = manager.invoke(messages)
        assert isinstance(result, AIMessage)

    def test_batch_method(self, dummy_chat_model):
        """Test batch method processes multiple inputs."""
        manager = MultiModelManager(models=[dummy_chat_model])

        # Batch process multiple inputs
        inputs = ["Message 1", "Message 2", "Message 3"]
        results = manager.batch(inputs)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, AIMessage)
            assert result.content == "This is a dummy response"

    def test_pipe_with_other_runnables(self, dummy_chat_model):
        """Test piping with other Runnable components."""
        manager = MultiModelManager(models=[dummy_chat_model])

        # Create a simple chain with pipe
        def uppercase_transform(message):
            if isinstance(message, AIMessage):
                return message.content.upper()
            return str(message).upper()

        from langchain_core.runnables import RunnableLambda

        uppercase_runnable = RunnableLambda(uppercase_transform)

        # Pipe manager output to uppercase transform
        chain = manager | uppercase_runnable

        result = chain.invoke("Test")
        assert result == "THIS IS A DUMMY RESPONSE"

    def test_chain_with_prompt_template(self, dummy_chat_model):
        """Test integration with LangChain prompt templates."""
        from langchain_core.prompts import ChatPromptTemplate

        manager = MultiModelManager(models=[dummy_chat_model])

        # Create a prompt template
        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are a helpful assistant."), ("human", "{input}")]
        )

        # Create chain
        chain = prompt | manager

        result = chain.invoke({"input": "Hello"})
        assert isinstance(result, AIMessage)

    def test_multiple_chains_with_same_manager(self, dummy_chat_model):
        """Test that same manager can be used in multiple chains."""
        manager = MultiModelManager(models=[dummy_chat_model])

        from langchain_core.prompts import ChatPromptTemplate

        # Create two different chains with the same manager
        prompt1 = ChatPromptTemplate.from_template("Question: {question}")
        chain1 = prompt1 | manager

        prompt2 = ChatPromptTemplate.from_template("Answer: {answer}")
        chain2 = prompt2 | manager

        # Both chains should work
        result1 = chain1.invoke({"question": "What is AI?"})
        result2 = chain2.invoke({"answer": "AI is..."})

        assert isinstance(result1, AIMessage)
        assert isinstance(result2, AIMessage)


# ============================================================================
# System Tests: Complex Scenarios
# ============================================================================


class TestComplexScenariosSystem:
    """System tests for complex real-world scenarios."""

    def test_priority_with_rate_limiting_and_fallback(self, multiple_dummy_models):
        """Test priority routing with rate limits and fallback."""
        configs = [
            ModelConfig(priority=10, max_rps=2),  # Highest priority, limited
            ModelConfig(priority=5, max_rps=2),  # Medium priority, limited
            ModelConfig(priority=1),  # Lowest priority, unlimited
        ]

        manager = MultiModelManager(
            models=multiple_dummy_models, model_configs=configs, strategy="priority"
        )

        # First 2 requests: model 0 (highest priority)
        result1 = manager.invoke("Test 1")
        result2 = manager.invoke("Test 2")
        assert "Response from model 1" in result1.content
        assert "Response from model 1" in result2.content

        # Next 2 requests: model 1 (fallback due to rate limit)
        result3 = manager.invoke("Test 3")
        result4 = manager.invoke("Test 4")
        assert "Response from model 2" in result3.content
        assert "Response from model 2" in result4.content

        # Next request: model 2 (both others rate limited)
        result5 = manager.invoke("Test 5")
        assert "Response from model 3" in result5.content

    def test_structured_output_with_routing_strategies(self, multiple_dummy_models):
        """Test structured output works with different routing strategies."""

        class Response(BaseModel):
            message: str
            count: int

        # Create models that return JSON
        from tests.conftest import DummyChatModel

        json_models = [
            DummyChatModel(
                model_name=f"json-{i}", response_text=f'{{"message": "Response {i}", "count": {i}}}'
            )
            for i in range(1, 4)
        ]

        # Test with round-robin strategy
        manager = MultiModelManager(models=json_models, strategy="round_robin")

        structured_chain = manager.with_structured_output(Response)

        # Make multiple requests
        results = []
        for _ in range(3):
            result = structured_chain.invoke("Get response")
            results.append(result)

        # Verify each model was used
        assert results[0].message == "Response 1"
        assert results[1].message == "Response 2"
        assert results[2].message == "Response 3"

    def test_batch_with_fallback(self, rate_limited_model, dummy_chat_model):
        """Test batch processing with fallback on failures."""
        manager = MultiModelManager(models=[rate_limited_model, dummy_chat_model])

        # Batch process - should fallback for all requests
        inputs = ["Message 1", "Message 2", "Message 3"]
        results = manager.batch(inputs)

        assert len(results) == 3
        for result in results:
            assert result.content == "This is a dummy response"

    def test_usage_tracking_across_requests(self, multiple_dummy_models):
        """Test that usage statistics are tracked correctly."""
        manager = MultiModelManager(models=multiple_dummy_models, strategy="round_robin")

        # Make several requests
        for i in range(6):
            manager.invoke(f"Test {i}")

        # Check usage statistics
        stats = manager._usage_tracker.get_all_stats()

        # Each model should have been used twice (6 requests / 3 models)
        assert stats[0].total_requests == 2
        assert stats[1].total_requests == 2
        assert stats[2].total_requests == 2

        # All should be successful
        assert stats[0].successful_requests == 2
        assert stats[1].successful_requests == 2
        assert stats[2].successful_requests == 2

    def test_mixed_success_and_failure_tracking(self, rate_limited_model, dummy_chat_model):
        """Test usage tracking with mixed success and failures."""
        manager = MultiModelManager(
            models=[rate_limited_model, dummy_chat_model], strategy="round_robin"
        )

        # Make multiple requests
        # First request: tries model 0 (fails), falls back to model 1 (succeeds)
        # Second request: tries model 1 (succeeds) - model 0 is in cooldown
        # Third request: tries model 1 (succeeds) - model 0 is still in cooldown
        for i in range(3):
            manager.invoke(f"Test {i}")

        stats = manager._usage_tracker.get_all_stats()

        # Model 0 should have 1 failed request (only tried once before cooldown)
        assert stats[0].total_requests == 1
        assert stats[0].failed_requests == 1
        assert stats[0].successful_requests == 0

        # Model 1 should have 3 successful requests (1 fallback + 2 direct)
        assert stats[1].total_requests == 3
        assert stats[1].successful_requests == 3
        assert stats[1].failed_requests == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
