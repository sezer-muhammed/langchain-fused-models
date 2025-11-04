"""Tests for Runnable interface methods in MultiModelManager."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_fused_model import MultiModelManager, ModelConfig


class TestInvokeMethod:
    """Tests for the invoke method (inherited from BaseChatModel)."""
    
    def test_invoke_with_string_input(self, dummy_chat_model):
        """Test invoke method with string input."""
        manager = MultiModelManager(models=[dummy_chat_model])
        
        result = manager.invoke("Hello, how are you?")
        
        assert result is not None
        assert hasattr(result, 'content')
        assert result.content == "This is a dummy response"
    
    def test_invoke_with_message_list(self, dummy_chat_model):
        """Test invoke method with list of messages."""
        manager = MultiModelManager(models=[dummy_chat_model])
        
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello!")
        ]
        result = manager.invoke(messages)
        
        assert result is not None
        assert hasattr(result, 'content')
        assert result.content == "This is a dummy response"
    
    def test_invoke_with_multiple_models(self, multiple_dummy_models):
        """Test invoke selects and uses a model correctly."""
        configs = [
            ModelConfig(priority=1),
            ModelConfig(priority=3),  # Highest priority
            ModelConfig(priority=2),
        ]
        
        manager = MultiModelManager(
            models=multiple_dummy_models,
            model_configs=configs,
            strategy="priority"
        )
        
        result = manager.invoke("Test message")
        
        # Should use model2 (highest priority)
        assert result.content == "Response from model 2"
    
    def test_invoke_with_kwargs(self, dummy_chat_model):
        """Test invoke method passes kwargs correctly."""
        manager = MultiModelManager(models=[dummy_chat_model])
        
        # Should not raise an error even with additional kwargs
        result = manager.invoke(
            "Hello",
            temperature=0.7,
            max_tokens=100
        )
        
        assert result is not None
        assert result.content == "This is a dummy response"


class TestBatchMethod:
    """Tests for the batch method (inherited from BaseChatModel)."""
    
    def test_batch_with_multiple_inputs(self, dummy_chat_model):
        """Test batch method with multiple string inputs."""
        manager = MultiModelManager(models=[dummy_chat_model])
        
        inputs = [
            "First message",
            "Second message",
            "Third message"
        ]
        
        results = manager.batch(inputs)
        
        assert len(results) == 3
        for result in results:
            assert hasattr(result, 'content')
            assert result.content == "This is a dummy response"
    
    def test_batch_with_message_lists(self, dummy_chat_model):
        """Test batch method with lists of messages."""
        manager = MultiModelManager(models=[dummy_chat_model])
        
        inputs = [
            [HumanMessage(content="First")],
            [HumanMessage(content="Second")],
            [HumanMessage(content="Third")]
        ]
        
        results = manager.batch(inputs)
        
        assert len(results) == 3
        for result in results:
            assert hasattr(result, 'content')
            assert result.content == "This is a dummy response"
    
    def test_batch_with_round_robin_strategy(self, multiple_dummy_models):
        """Test batch distributes requests across models with round-robin."""
        manager = MultiModelManager(
            models=multiple_dummy_models,
            strategy="round_robin"
        )
        
        inputs = ["Message 1", "Message 2", "Message 3"]
        results = manager.batch(inputs)
        
        assert len(results) == 3
        # With round-robin, each model should be used once
        assert results[0].content == "Response from model 1"
        assert results[1].content == "Response from model 2"
        assert results[2].content == "Response from model 3"
    
    def test_batch_empty_list(self, dummy_chat_model):
        """Test batch method with empty input list."""
        manager = MultiModelManager(models=[dummy_chat_model])
        
        results = manager.batch([])
        
        assert results == []


class TestPipeMethod:
    """Tests for the pipe method for chaining with other Runnables."""
    
    def test_pipe_with_output_parser(self, dummy_chat_model):
        """Test piping manager with an output parser."""
        manager = MultiModelManager(models=[dummy_chat_model])
        parser = StrOutputParser()
        
        # Create a chain: manager | parser
        chain = manager | parser
        
        result = chain.invoke("Hello")
        
        # Parser should extract the string content
        assert isinstance(result, str)
        assert result == "This is a dummy response"
    
    def test_pipe_with_prompt_template(self, dummy_chat_model):
        """Test piping prompt template with manager."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant"),
            ("human", "{input}")
        ])
        manager = MultiModelManager(models=[dummy_chat_model])
        
        # Create a chain: prompt | manager
        chain = prompt | manager
        
        result = chain.invoke({"input": "Hello"})
        
        assert hasattr(result, 'content')
        assert result.content == "This is a dummy response"
    
    def test_pipe_full_chain(self, dummy_chat_model):
        """Test a full chain with prompt, manager, and parser."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant"),
            ("human", "{question}")
        ])
        manager = MultiModelManager(models=[dummy_chat_model])
        parser = StrOutputParser()
        
        # Create full chain: prompt | manager | parser
        chain = prompt | manager | parser
        
        result = chain.invoke({"question": "What is 2+2?"})
        
        assert isinstance(result, str)
        assert result == "This is a dummy response"
    
    def test_pipe_with_passthrough(self, dummy_chat_model):
        """Test piping with RunnablePassthrough."""
        manager = MultiModelManager(models=[dummy_chat_model])
        
        # Create a chain that passes through and then invokes
        chain = RunnablePassthrough() | manager
        
        result = chain.invoke("Test input")
        
        assert hasattr(result, 'content')
        assert result.content == "This is a dummy response"


class TestLangChainIntegration:
    """Tests for integration with LangChain chains and components."""
    
    def test_manager_in_sequential_chain(self, multiple_dummy_models):
        """Test manager works in a sequential chain."""
        prompt1 = ChatPromptTemplate.from_template("First step: {input}")
        manager1 = MultiModelManager(models=[multiple_dummy_models[0]])
        parser = StrOutputParser()
        
        # First chain
        chain1 = prompt1 | manager1 | parser
        
        result = chain1.invoke({"input": "test"})
        
        assert isinstance(result, str)
        assert result == "Response from model 1"
    
    def test_manager_with_different_strategies_in_chain(self, multiple_dummy_models):
        """Test managers with different strategies can be chained."""
        manager_priority = MultiModelManager(
            models=multiple_dummy_models,
            model_configs=[
                ModelConfig(priority=1),
                ModelConfig(priority=3),
                ModelConfig(priority=2)
            ],
            strategy="priority"
        )
        
        parser = StrOutputParser()
        chain = manager_priority | parser
        
        result = chain.invoke("Test")
        
        # Should use highest priority model (model2)
        assert result == "Response from model 2"
    
    def test_manager_preserves_langchain_behavior(self, dummy_chat_model):
        """Test that manager behaves like a standard ChatModel in chains."""
        manager = MultiModelManager(models=[dummy_chat_model])
        
        # Test that it has the same interface as BaseChatModel
        assert hasattr(manager, 'invoke')
        assert hasattr(manager, 'batch')
        assert hasattr(manager, 'stream')
        assert hasattr(manager, 'pipe')
        assert hasattr(manager, '_generate')
        assert hasattr(manager, '_llm_type')
    
    def test_manager_with_complex_chain(self, dummy_chat_model):
        """Test manager in a more complex chain with multiple steps."""
        # Create a chain that processes input through multiple steps
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a {role}"),
            ("human", "{input}")
        ])
        manager = MultiModelManager(models=[dummy_chat_model])
        parser = StrOutputParser()
        
        chain = (
            {"role": lambda x: "helpful assistant", "input": RunnablePassthrough()}
            | prompt
            | manager
            | parser
        )
        
        result = chain.invoke("Hello")
        
        assert isinstance(result, str)
        assert result == "This is a dummy response"
    
    def test_manager_invoke_returns_aimessage(self, dummy_chat_model):
        """Test that invoke returns an AIMessage compatible object."""
        manager = MultiModelManager(models=[dummy_chat_model])
        
        result = manager.invoke("Test")
        
        # Should return an AIMessage
        assert hasattr(result, 'content')
        assert hasattr(result, 'type')
        assert result.type == 'ai'
    
    def test_manager_batch_returns_list_of_aimessages(self, dummy_chat_model):
        """Test that batch returns a list of AIMessage compatible objects."""
        manager = MultiModelManager(models=[dummy_chat_model])
        
        results = manager.batch(["Test 1", "Test 2"])
        
        assert len(results) == 2
        for result in results:
            assert hasattr(result, 'content')
            assert hasattr(result, 'type')
            assert result.type == 'ai'


class TestRunnableWithFallback:
    """Tests for Runnable interface with fallback behavior."""
    
    def test_invoke_with_fallback_on_error(self, rate_limited_model, dummy_chat_model):
        """Test invoke falls back to next model on error."""
        manager = MultiModelManager(
            models=[rate_limited_model, dummy_chat_model],
            default_fallback=True
        )
        
        result = manager.invoke("Test")
        
        # Should fallback to dummy model
        assert result.content == "This is a dummy response"
    
    def test_batch_with_fallback(self, rate_limited_model, dummy_chat_model):
        """Test batch operations with fallback."""
        manager = MultiModelManager(
            models=[rate_limited_model, dummy_chat_model],
            default_fallback=True
        )
        
        results = manager.batch(["Test 1", "Test 2"])
        
        assert len(results) == 2
        for result in results:
            assert result.content == "This is a dummy response"
    
    def test_pipe_with_fallback(self, rate_limited_model, dummy_chat_model):
        """Test piped chain with fallback behavior."""
        manager = MultiModelManager(
            models=[rate_limited_model, dummy_chat_model],
            default_fallback=True
        )
        parser = StrOutputParser()
        
        chain = manager | parser
        result = chain.invoke("Test")
        
        assert result == "This is a dummy response"
