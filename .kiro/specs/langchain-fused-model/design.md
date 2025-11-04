# Design Document

## Overview

The langchain-fused-model package provides a `MultiModelManager` class that acts as a unified interface for managing multiple LangChain ChatModel instances. The design follows LangChain's architecture patterns by inheriting from `BaseChatModel` and implementing the Runnable interface, ensuring seamless integration with existing LangChain workflows.

The system uses a modular architecture with separate components for routing strategies, rate limiting, usage tracking, and structured output handling. This separation of concerns enables extensibility and maintainability.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  MultiModelManager                       │
│  (BaseChatModel, Runnable)                              │
├─────────────────────────────────────────────────────────┤
│  - Model Registry                                        │
│  - Strategy Selector                                     │
│  - Rate Limiter                                          │
│  - Usage Tracker                                         │
│  - Structured Output Handler                            │
└─────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ ChatModel 1  │    │ ChatModel 2  │    │ ChatModel 3  │
│ (OpenAI)     │    │ (Anthropic)  │    │ (Local)      │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Package Structure

```
langchain-fused-model/
├── src/
│   └── langchain_fused_model/
│       ├── __init__.py
│       ├── manager.py              # MultiModelManager class
│       ├── strategy.py             # Routing strategies
│       ├── rate_limiter.py         # Rate limiting logic
│       ├── usage_tracker.py        # Usage statistics
│       ├── structured_output.py    # Structured output handling
│       └── exceptions.py           # Custom exceptions
├── tests/
│   ├── __init__.py
│   ├── test_manager.py
│   ├── test_strategies.py
│   ├── test_rate_limiter.py
│   ├── test_structured_output.py
│   └── conftest.py                 # Pytest fixtures
├── examples/
│   ├── basic_usage.ipynb
│   ├── routing_strategies.ipynb
│   └── structured_output.ipynb
├── pyproject.toml
├── README.md
├── LICENSE
├── CONTRIBUTING.md
└── .github/
    └── workflows/
        ├── test.yml
        └── lint.yml
```

## Components and Interfaces

### 1. MultiModelManager

**Purpose**: Main class that manages multiple ChatModel instances and routes requests.

**Key Attributes**:
- `models: List[ChatModel]` - List of managed ChatModel instances
- `model_configs: List[ModelConfig]` - Configuration for each model (priority, rate limits, cost)
- `strategy: RoutingStrategy` - Current routing strategy
- `usage_tracker: UsageTracker` - Tracks usage statistics per model
- `rate_limiter: RateLimiter` - Manages rate limits per model

**Key Methods**:
```python
class MultiModelManager(BaseChatModel):
    def __init__(
        self,
        models: List[ChatModel],
        model_configs: Optional[List[ModelConfig]] = None,
        strategy: Union[RoutingStrategy, Callable] = RoutingStrategy.PRIORITY,
        default_fallback: bool = True
    ):
        """Initialize the manager with models and configuration."""
        
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        """Generate response using selected model with fallback."""
        
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "multi-model-manager"
        
    def with_structured_output(
        self,
        schema: Type[BaseModel],
        **kwargs: Any
    ) -> Runnable:
        """Return a runnable that outputs structured data."""
        
    def _select_model(self) -> Tuple[ChatModel, int]:
        """Select model based on strategy and availability."""
        
    def _invoke_with_fallback(
        self,
        model_idx: int,
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> ChatResult:
        """Invoke model with automatic fallback on failure."""
```

### 2. ModelConfig

**Purpose**: Configuration data class for each model.

```python
@dataclass
class ModelConfig:
    priority: int = 0                    # Higher = higher priority
    max_rpm: Optional[int] = None        # Max requests per minute
    max_rps: Optional[int] = None        # Max requests per second
    cost_per_1k_tokens: float = 0.0      # Cost for pricing strategy
    timeout: Optional[float] = None      # Request timeout
    retry_on_errors: List[Type[Exception]] = field(default_factory=list)
```

### 3. RoutingStrategy

**Purpose**: Enum and implementations for different routing strategies.

```python
class RoutingStrategy(Enum):
    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"
    LEAST_USED = "least_used"
    COST_AWARE = "cost_aware"

class StrategySelector:
    """Selects model based on configured strategy."""
    
    def select(
        self,
        strategy: RoutingStrategy,
        models: List[ChatModel],
        configs: List[ModelConfig],
        usage_stats: Dict[int, UsageStats],
        available_models: List[int]
    ) -> int:
        """Return index of selected model."""
        
    def _priority_select(self, ...) -> int:
        """Select highest priority available model."""
        
    def _round_robin_select(self, ...) -> int:
        """Select next model in rotation."""
        
    def _least_used_select(self, ...) -> int:
        """Select model with lowest usage count."""
        
    def _cost_aware_select(self, ...) -> int:
        """Select lowest cost available model."""
```

### 4. RateLimiter

**Purpose**: Manages rate limits and cooldowns for each model.

```python
class RateLimiter:
    """Tracks and enforces rate limits per model."""
    
    def __init__(self):
        self._request_times: Dict[int, List[float]] = {}
        self._cooldowns: Dict[int, float] = {}
        
    def is_available(self, model_idx: int, config: ModelConfig) -> bool:
        """Check if model is available (not rate limited)."""
        
    def record_request(self, model_idx: int) -> None:
        """Record a request timestamp for rate tracking."""
        
    def set_cooldown(self, model_idx: int, duration: float) -> None:
        """Set cooldown period for a model."""
        
    def _check_rpm(self, model_idx: int, max_rpm: int) -> bool:
        """Check requests per minute limit."""
        
    def _check_rps(self, model_idx: int, max_rps: int) -> bool:
        """Check requests per second limit."""
```

### 5. UsageTracker

**Purpose**: Tracks usage statistics for each model.

```python
@dataclass
class UsageStats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    last_used: Optional[float] = None

class UsageTracker:
    """Tracks usage statistics per model."""
    
    def __init__(self):
        self._stats: Dict[int, UsageStats] = {}
        
    def record_request(
        self,
        model_idx: int,
        success: bool,
        tokens: int = 0
    ) -> None:
        """Record request outcome and token usage."""
        
    def get_stats(self, model_idx: int) -> UsageStats:
        """Get usage statistics for a model."""
        
    def get_all_stats(self) -> Dict[int, UsageStats]:
        """Get statistics for all models."""
```

### 6. StructuredOutputHandler

**Purpose**: Handles structured output with native support detection and fallback.

```python
class StructuredOutputHandler:
    """Manages structured output for models."""
    
    def create_structured_runnable(
        self,
        manager: MultiModelManager,
        schema: Type[BaseModel],
        **kwargs: Any
    ) -> Runnable:
        """Create a runnable that returns structured output."""
        
    def has_native_support(self, model: ChatModel) -> bool:
        """Check if model has native structured output support."""
        
    def invoke_native(
        self,
        model: ChatModel,
        schema: Type[BaseModel],
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> BaseModel:
        """Invoke model with native structured output."""
        
    def invoke_with_parsing(
        self,
        model: ChatModel,
        schema: Type[BaseModel],
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> BaseModel:
        """Invoke model and parse output to structured format."""
        
    def _inject_format_instructions(
        self,
        messages: List[BaseMessage],
        schema: Type[BaseModel]
    ) -> List[BaseMessage]:
        """Add JSON format instructions to messages."""
        
    def _extract_json(self, text: str) -> str:
        """Extract JSON from model response using regex."""
        
    def _parse_to_model(self, json_str: str, schema: Type[BaseModel]) -> BaseModel:
        """Parse JSON string to Pydantic model."""
```

### 7. Custom Exceptions

```python
class MultiModelError(Exception):
    """Base exception for multi-model errors."""

class AllModelsFailedError(MultiModelError):
    """Raised when all models fail to respond."""
    
class RateLimitExceededError(MultiModelError):
    """Raised when all models are rate limited."""
    
class StructuredOutputError(MultiModelError):
    """Raised when structured output parsing fails."""
```

## Data Models

### ModelConfig Schema

```python
{
    "priority": int,              # 0-100, higher = more preferred
    "max_rpm": Optional[int],     # Maximum requests per minute
    "max_rps": Optional[int],     # Maximum requests per second
    "cost_per_1k_tokens": float,  # Cost in dollars per 1k tokens
    "timeout": Optional[float],   # Request timeout in seconds
    "retry_on_errors": List[Type[Exception]]  # Exception types to retry
}
```

### Usage Statistics Schema

```python
{
    "total_requests": int,
    "successful_requests": int,
    "failed_requests": int,
    "total_tokens": int,
    "last_used": Optional[float]  # Unix timestamp
}
```

## Error Handling

### Error Hierarchy

1. **Rate Limit Errors**: Caught and trigger model switching
2. **Timeout Errors**: Caught and trigger model switching
3. **API Errors**: Logged and trigger fallback based on configuration
4. **All Models Failed**: Raise `AllModelsFailedError` with details
5. **Structured Output Errors**: Raise `StructuredOutputError` with parsing details

### Fallback Logic

```
1. Select model using strategy
2. Check rate limit availability
3. Attempt request with selected model
4. On failure:
   a. Log error and model used
   b. Mark model as temporarily unavailable (if rate limit)
   c. Select next available model
   d. Retry with new model
5. If all models exhausted:
   a. Raise AllModelsFailedError with all error details
```

### Logging Strategy

- Log model selection decisions (DEBUG level)
- Log rate limit events (INFO level)
- Log fallback events (WARNING level)
- Log all model failures (ERROR level)
- Include model identifier in all log messages

## Testing Strategy

### Unit Tests

1. **Manager Tests** (`test_manager.py`):
   - Test initialization with various configurations
   - Test model selection with different strategies
   - Test fallback behavior on errors
   - Test LangChain integration (invoke, batch, pipe)

2. **Strategy Tests** (`test_strategies.py`):
   - Test PRIORITY strategy selection
   - Test ROUND_ROBIN distribution
   - Test LEAST_USED selection
   - Test COST_AWARE selection
   - Test custom strategy functions

3. **Rate Limiter Tests** (`test_rate_limiter.py`):
   - Test RPM limit enforcement
   - Test RPS limit enforcement
   - Test cooldown behavior
   - Test availability checking

4. **Structured Output Tests** (`test_structured_output.py`):
   - Test native support detection
   - Test native structured output delegation
   - Test fallback prompt injection
   - Test JSON extraction from responses
   - Test Pydantic validation
   - Test error handling for invalid JSON

### Test Fixtures

```python
# conftest.py
@pytest.fixture
def dummy_chat_model():
    """Create a mock ChatModel for testing."""
    
@pytest.fixture
def rate_limited_model():
    """Create a model that raises rate limit errors."""
    
@pytest.fixture
def timeout_model():
    """Create a model that raises timeout errors."""
    
@pytest.fixture
def structured_output_model():
    """Create a model with native structured output support."""
```

### Integration Tests

- Test with real LangChain chains
- Test with LangChain agents
- Test callback emission
- Test streaming support (if implemented)

## LangChain Integration Details

### BaseChatModel Implementation

The `MultiModelManager` must implement:
- `_generate()`: Core generation method
- `_llm_type`: Property returning model type identifier
- `_identifying_params`: Property for model identification

### Runnable Interface

Inherited from `BaseChatModel`, provides:
- `invoke()`: Single invocation
- `batch()`: Batch invocations
- `stream()`: Streaming (optional)
- `pipe()`: Chaining with other runnables

### Callback Integration

```python
# In _generate method
if run_manager:
    run_manager.on_llm_start(...)
    
try:
    result = model.generate(...)
    if run_manager:
        run_manager.on_llm_end(...)
except Exception as e:
    if run_manager:
        run_manager.on_llm_error(e)
    raise
```

## Performance Considerations

1. **Rate Limit Tracking**: Use efficient time-window tracking with deque
2. **Model Selection**: Cache strategy decisions when possible
3. **Thread Safety**: Use locks for shared state (usage tracker, rate limiter)
4. **Memory**: Limit stored request timestamps to relevant time windows

## Security Considerations

1. **API Keys**: Never log or expose API keys from underlying models
2. **Input Validation**: Validate all configuration inputs
3. **Error Messages**: Sanitize error messages to avoid leaking sensitive data
4. **Dependencies**: Pin dependency versions for security

## Extensibility Points

1. **Custom Strategies**: Accept callable for custom routing logic
2. **Custom Error Handlers**: Allow registration of error handlers
3. **Middleware**: Support pre/post processing hooks
4. **Metrics**: Provide hooks for external metrics collection
