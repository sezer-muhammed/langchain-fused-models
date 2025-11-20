"""Pytest fixtures for testing langchain-fused-model."""

from typing import Any, List, Optional, Type

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel


class DummyChatModel(BaseChatModel):
    """A simple mock ChatModel for testing."""

    model_name: str = "dummy"
    response_text: str = "This is a dummy response"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a dummy response."""
        message = AIMessage(content=self.response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return f"dummy-{self.model_name}"

    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters."""
        return {"model_name": self.model_name}


class RateLimitedChatModel(BaseChatModel):
    """A mock ChatModel that raises rate limit errors."""

    model_name: str = "rate-limited"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Raise a rate limit error."""
        raise Exception("Rate limit exceeded")

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return f"rate-limited-{self.model_name}"

    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters."""
        return {"model_name": self.model_name}


class TimeoutChatModel(BaseChatModel):
    """A mock ChatModel that raises timeout errors."""

    model_name: str = "timeout"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Raise a timeout error."""
        raise Exception("Request timeout")

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return f"timeout-{self.model_name}"

    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters."""
        return {"model_name": self.model_name}


class StructuredOutputChatModel(BaseChatModel):
    """A mock ChatModel with native structured output support."""

    model_name: str = "structured"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response."""
        message = AIMessage(content='{"name": "John", "age": 30}')
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any):
        """Mock native structured output support."""

        def structured_invoke(messages):
            # Return a mock instance of the schema
            if hasattr(schema, "model_validate"):
                return schema.model_validate({"name": "John", "age": 30})
            else:
                return schema(name="John", age=30)

        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(structured_invoke)

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return f"structured-{self.model_name}"

    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters."""
        return {"model_name": self.model_name}


@pytest.fixture
def dummy_chat_model():
    """Create a dummy ChatModel for testing."""
    return DummyChatModel()


@pytest.fixture
def rate_limited_model():
    """Create a model that raises rate limit errors."""
    return RateLimitedChatModel()


@pytest.fixture
def timeout_model():
    """Create a model that raises timeout errors."""
    return TimeoutChatModel()


@pytest.fixture
def structured_output_model():
    """Create a model with native structured output support."""
    return StructuredOutputChatModel()


@pytest.fixture
def multiple_dummy_models():
    """Create multiple dummy models with different responses."""
    return [
        DummyChatModel(model_name="model1", response_text="Response from model 1"),
        DummyChatModel(model_name="model2", response_text="Response from model 2"),
        DummyChatModel(model_name="model3", response_text="Response from model 3"),
    ]
