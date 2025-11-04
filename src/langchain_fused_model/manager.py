"""Core MultiModelManager class for managing multiple ChatModel instances."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Type, Any, Union, Callable, Tuple
from pydantic import BaseModel, PrivateAttr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration, LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.runnables import Runnable, RunnableLambda

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a managed ChatModel instance.
    
    Attributes:
        priority: Priority level for the model (higher = more preferred). Default is 0.
        max_rpm: Maximum requests per minute allowed for this model.
        max_rps: Maximum requests per second allowed for this model.
        cost_per_1k_tokens: Cost in dollars per 1000 tokens for pricing strategy.
        timeout: Request timeout in seconds.
        retry_on_errors: List of exception types that should trigger retry/fallback.
    """
    priority: int = 0
    max_rpm: Optional[int] = None
    max_rps: Optional[int] = None
    cost_per_1k_tokens: float = 0.0
    timeout: Optional[float] = None
    retry_on_errors: List[Type[Exception]] = field(default_factory=list)


class MultiModelManager(BaseChatModel):
    """Manages multiple ChatModel instances with intelligent routing and fallback.
    
    This class provides a unified interface for managing multiple LangChain ChatModel
    instances with features like:
    - Intelligent routing strategies (priority, round-robin, least-used, cost-aware)
    - Automatic rate limiting per model
    - Automatic fallback on errors
    - Usage tracking and statistics
    - Full LangChain Runnable compatibility
    
    The manager inherits from BaseChatModel, making it a drop-in replacement for
    any LangChain ChatModel in chains, agents, and other workflows.
    """
    
    # Declare fields for Pydantic v2 compatibility
    models: List[BaseChatModel]
    model_configs: List[ModelConfig]
    strategy: Union[str, Callable]
    default_fallback: bool
    
    # Private attributes for internal components (not part of Pydantic model)
    _usage_tracker: Any = PrivateAttr()
    _rate_limiter: Any = PrivateAttr()
    _strategy_selector: Any = PrivateAttr()
    
    # Use model_config to allow arbitrary types (for ChatModel instances)
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(
        self,
        models: List[BaseChatModel],
        model_configs: Optional[List[ModelConfig]] = None,
        strategy: Union[str, Callable] = "priority",
        default_fallback: bool = True,
        **kwargs: Any
    ):
        """Initialize the MultiModelManager.
        
        Args:
            models: List of ChatModel instances to manage.
            model_configs: Optional list of ModelConfig objects, one per model.
                If not provided, default configurations will be created.
            strategy: Routing strategy to use. Can be a string ("priority",
                "round_robin", "least_used", "cost_aware") or a custom callable.
                Default is "priority".
            default_fallback: Whether to enable automatic fallback on errors.
                Default is True.
            **kwargs: Additional keyword arguments for BaseChatModel.
        
        Raises:
            ValueError: If models list is empty or if model_configs length
                doesn't match models length.
        """
        if not models:
            raise ValueError("At least one model must be provided")
        
        # Create default configs if not provided
        if model_configs is None:
            model_configs = [ModelConfig() for _ in models]
        else:
            if len(model_configs) != len(models):
                raise ValueError(
                    f"Number of model_configs ({len(model_configs)}) must match "
                    f"number of models ({len(models)})"
                )
        
        # Initialize parent class with field values
        super().__init__(
            models=models,
            model_configs=model_configs,
            strategy=strategy,
            default_fallback=default_fallback,
            **kwargs
        )
        
        # Initialize internal components (private attributes)
        from .usage_tracker import UsageTracker
        from .rate_limiter import RateLimiter
        from .strategy import StrategySelector
        
        self._usage_tracker = UsageTracker()
        self._rate_limiter = RateLimiter()
        self._strategy_selector = StrategySelector()
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type.
        
        Returns:
            String identifier "multi-model-manager".
        """
        return "multi-model-manager"
    
    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters for this manager.
        
        Returns:
            Dictionary containing identifying information about the manager
            and its managed models.
        """
        return {
            "manager_type": "multi-model-manager",
            "num_models": len(self.models),
            "strategy": str(self.strategy),
            "model_types": [model._llm_type for model in self.models],
        }
    
    def _select_model(self) -> Tuple[BaseChatModel, int]:
        """Select a model based on strategy and availability.
        
        This method filters available models based on rate limits and cooldowns,
        then uses the configured routing strategy to select the best model.
        
        Returns:
            Tuple of (selected ChatModel instance, model index)
        
        Raises:
            RateLimitExceededError: If all models are rate limited or unavailable.
        """
        from .exceptions import RateLimitExceededError
        
        # Filter available models based on rate limits
        available_models = []
        for idx, config in enumerate(self.model_configs):
            if self._rate_limiter.is_available(idx, config):
                available_models.append(idx)
        
        # Check if any models are available
        if not available_models:
            logger.warning("All models are currently rate limited or unavailable")
            raise RateLimitExceededError("All models are currently rate limited or unavailable")
        
        # Get usage statistics for strategy selection
        usage_stats = self._usage_tracker.get_all_stats()
        
        # Use strategy selector to choose model
        selected_idx = self._strategy_selector.select(
            strategy=self.strategy,
            models=self.models,
            configs=self.model_configs,
            usage_stats=usage_stats,
            available_models=available_models
        )
        
        selected_model = self.models[selected_idx]
        
        logger.debug(
            f"Selected model {selected_idx} (type: {selected_model._llm_type}) "
            f"using strategy {self.strategy}"
        )
        
        return selected_model, selected_idx
    
    def _invoke_with_fallback(
        self,
        model_idx: int,
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> ChatResult:
        """Invoke model with automatic fallback on failure.
        
        This method attempts to invoke the specified model and automatically
        falls back to other available models if the request fails due to
        rate limits, timeouts, or other configured errors.
        
        Args:
            model_idx: Index of the initial model to try.
            messages: List of messages to send to the model.
            **kwargs: Additional keyword arguments to pass to the model.
        
        Returns:
            ChatResult from the successful model invocation.
        
        Raises:
            AllModelsFailedError: If all models fail to respond.
        """
        from .exceptions import AllModelsFailedError
        
        # Track which models we've tried and their errors
        tried_models = set()
        errors = {}
        
        current_idx = model_idx
        
        while current_idx not in tried_models:
            tried_models.add(current_idx)
            model = self.models[current_idx]
            config = self.model_configs[current_idx]
            
            try:
                logger.debug(f"Attempting request with model {current_idx} (type: {model._llm_type})")
                
                # Record the request attempt
                self._rate_limiter.record_request(current_idx)
                
                # Invoke the model directly with _generate
                result = model._generate(
                    messages=messages,
                    **kwargs
                )
                
                # Record successful request
                tokens = 0
                if hasattr(result, 'llm_output') and result.llm_output and 'token_usage' in result.llm_output:
                    tokens = result.llm_output['token_usage'].get('total_tokens', 0)
                
                self._usage_tracker.record_request(current_idx, success=True, tokens=tokens)
                
                logger.info(f"Successfully completed request with model {current_idx}")
                
                return result
                
            except Exception as e:
                # Record failed request
                self._usage_tracker.record_request(current_idx, success=False)
                
                # Store error details
                errors[current_idx] = {
                    "model_type": model._llm_type,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
                
                logger.warning(
                    f"Model {current_idx} failed with {type(e).__name__}: {str(e)}"
                )
                
                # Check if this error type should trigger retry
                should_retry = False
                
                # Check for rate limit errors (common patterns)
                error_name = type(e).__name__.lower()
                error_msg = str(e).lower()
                if 'rate' in error_name or 'rate' in error_msg or 'limit' in error_msg:
                    logger.info(f"Rate limit detected for model {current_idx}, setting cooldown")
                    self._rate_limiter.set_cooldown(current_idx, 60.0)  # 60 second cooldown
                    should_retry = True
                
                # Check for timeout errors
                elif 'timeout' in error_name or 'timeout' in error_msg:
                    logger.info(f"Timeout detected for model {current_idx}")
                    should_retry = True
                
                # Check configured retry errors
                elif config.retry_on_errors:
                    for error_type in config.retry_on_errors:
                        if isinstance(e, error_type):
                            logger.info(f"Configured retry error detected for model {current_idx}")
                            should_retry = True
                            break
                
                # If fallback is disabled or we shouldn't retry, re-raise
                if not self.default_fallback or not should_retry:
                    logger.error(f"Fallback disabled or error not retryable, re-raising exception")
                    raise
                
                # Try to select another model
                try:
                    logger.info(f"Attempting fallback from model {current_idx}")
                    next_model, current_idx = self._select_model()
                    
                    # If we've already tried this model, we're out of options
                    if current_idx in tried_models:
                        logger.error("No more untried models available")
                        break
                    
                except Exception as select_error:
                    # No more models available
                    logger.error(f"Failed to select fallback model: {select_error}")
                    break
        
        # All models have been tried and failed
        logger.error(f"All models failed after trying {len(tried_models)} model(s)")
        raise AllModelsFailedError(
            errors=errors,
            message=f"All {len(tried_models)} model(s) failed to respond. Errors: {errors}"
        )
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        """Generate response using selected model with fallback.
        
        This is the core method required by BaseChatModel. It handles model
        selection, invocation with fallback, usage tracking, and callback
        emission for LangChain integration.
        
        Args:
            messages: List of messages to send to the model.
            stop: Optional list of stop sequences.
            run_manager: Optional callback manager for LLM run events.
            **kwargs: Additional keyword arguments to pass to the model.
        
        Returns:
            ChatResult containing the generated response.
        
        Raises:
            AllModelsFailedError: If all models fail to respond.
            RateLimitExceededError: If all models are rate limited.
        """
        # Prepare kwargs for model invocation
        if stop is not None:
            kwargs['stop'] = stop
        
        try:
            # Select initial model based on strategy and availability
            selected_model, model_idx = self._select_model()
            
            logger.debug(
                f"Starting generation with model {model_idx} "
                f"(type: {selected_model._llm_type})"
            )
            
            # Invoke with automatic fallback support
            result = self._invoke_with_fallback(
                model_idx=model_idx,
                messages=messages,
                **kwargs
            )
            
            logger.info("Generation completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed with error: {type(e).__name__}: {str(e)}")
            
            # Re-raise the exception
            raise
    
    def with_structured_output(
        self,
        schema: Type[BaseModel],
        **kwargs: Any
    ) -> Runnable:
        """Return a runnable that outputs structured data.
        
        This method creates a Runnable that wraps the MultiModelManager and
        applies structured output logic. It automatically detects whether the
        selected model has native structured output support and uses the
        appropriate method (native or fallback parsing).
        
        Args:
            schema: Pydantic model class defining the expected output structure.
            **kwargs: Additional keyword arguments to pass to the structured
                output handler or underlying model.
        
        Returns:
            A Runnable that takes messages as input and returns an instance
            of the Pydantic model with validated structured data.
        
        Example:
            >>> from pydantic import BaseModel
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>> 
            >>> manager = MultiModelManager(models=[...])
            >>> structured_chain = manager.with_structured_output(Person)
            >>> result = structured_chain.invoke("Tell me about John who is 30")
            >>> print(result.name, result.age)  # "John" 30
        """
        from .structured_output import StructuredOutputHandler
        
        # Create handler instance
        handler = StructuredOutputHandler()
        
        # Define the structured output function
        def structured_output_func(messages: Union[List[BaseMessage], str]) -> BaseModel:
            """Internal function that handles structured output logic.
            
            Args:
                messages: Input messages (list of BaseMessage or string).
            
            Returns:
                Instance of the Pydantic model with parsed data.
            """
            # Convert string input to message list if needed
            if isinstance(messages, str):
                from langchain_core.messages import HumanMessage
                messages = [HumanMessage(content=messages)]
            
            # Select a model based on strategy and availability
            selected_model, model_idx = self._select_model()
            
            logger.debug(
                f"Selected model {model_idx} for structured output "
                f"(schema: {schema.__name__})"
            )
            
            # Check if the selected model has native structured output support
            if handler.has_native_support(selected_model):
                logger.debug(f"Using native structured output for model {model_idx}")
                
                try:
                    # Use native structured output
                    result = handler.invoke_native(
                        model=selected_model,
                        schema=schema,
                        messages=messages,
                        **kwargs
                    )
                    
                    # Record successful request
                    self._usage_tracker.record_request(model_idx, success=True)
                    self._rate_limiter.record_request(model_idx)
                    
                    return result
                    
                except Exception as e:
                    # Record failed request
                    self._usage_tracker.record_request(model_idx, success=False)
                    
                    logger.warning(
                        f"Native structured output failed for model {model_idx}: "
                        f"{type(e).__name__}: {str(e)}"
                    )
                    
                    # If fallback is enabled, try fallback parsing
                    if self.default_fallback:
                        logger.info(f"Attempting fallback parsing for model {model_idx}")
                        
                        try:
                            result = handler.invoke_with_parsing(
                                model=selected_model,
                                schema=schema,
                                messages=messages,
                                **kwargs
                            )
                            
                            # Record successful request
                            self._usage_tracker.record_request(model_idx, success=True)
                            
                            return result
                            
                        except Exception as fallback_error:
                            logger.error(
                                f"Fallback parsing also failed: "
                                f"{type(fallback_error).__name__}: {str(fallback_error)}"
                            )
                            raise
                    else:
                        # Re-raise if fallback is disabled
                        raise
            else:
                logger.debug(f"Using fallback parsing for model {model_idx}")
                
                try:
                    # Use fallback parsing
                    result = handler.invoke_with_parsing(
                        model=selected_model,
                        schema=schema,
                        messages=messages,
                        **kwargs
                    )
                    
                    # Record successful request
                    self._usage_tracker.record_request(model_idx, success=True)
                    self._rate_limiter.record_request(model_idx)
                    
                    return result
                    
                except Exception as e:
                    # Record failed request
                    self._usage_tracker.record_request(model_idx, success=False)
                    
                    logger.error(
                        f"Fallback parsing failed for model {model_idx}: "
                        f"{type(e).__name__}: {str(e)}"
                    )
                    raise
        
        # Wrap the function in a RunnableLambda to make it a proper Runnable
        return RunnableLambda(structured_output_func)
