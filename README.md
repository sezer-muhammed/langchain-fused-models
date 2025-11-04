# langchain-fused-model

Intelligent routing and management for multiple LangChain ChatModel instances with advanced features like rate limiting, automatic fallback, and structured output support.

## Table of Contents

- [Overview](#overview)
- [Why langchain-fused-model](#why-langchain-fused-model)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Routing Strategies](#routing-strategies)
  - [Priority-Based Routing](#priority-based-routing)
  - [Cost-Aware Routing](#cost-aware-routing)
  - [Round-Robin Routing](#round-robin-routing)
  - [Least-Used Routing](#least-used-routing)
  - [Custom Strategies](#custom-strategies)
- [Structured Output](#structured-output)
- [Rate Limiting and Fallback](#rate-limiting-and-fallback)
- [LangChain Integration](#langchain-integration)
- [Usage Statistics](#usage-statistics)
- [Advanced Configuration](#advanced-configuration)
- [Examples](#examples)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

---

## Overview

`langchain-fused-model` provides a `MultiModelManager` class that acts as a unified interface for managing multiple LangChain ChatModel instances. It enables dynamic model selection based on configurable strategies while maintaining full LangChain compatibility.

The manager inherits from LangChain's `BaseChatModel`, making it a drop-in replacement for any ChatModel in chains, agents, and other workflows.

## Why langchain-fused-model

Many developers today rely on multiple large language model providers to balance cost, availability, latency, and capabilities. However, LangChain does not provide a unified interface to dynamically route across multiple models based on rate limits or priorities. This project was created to fill that gap.

Whether you're managing free-tier APIs, orchestrating across OpenAI and Anthropic, or experimenting with cost-based strategies, `langchain-fused-model` helps you:

- **Fail gracefully** when APIs are throttled or down
- **Reduce latency or cost** by routing requests optimally
- **Extract structured outputs** even from models that don't support it natively
- **Scale production chains and agents** with built-in observability and fallback

## Features

- **Multiple Routing Strategies**: Priority-based, round-robin, least-used, and cost-aware routing
- **Automatic Rate Limiting**: Per-model rate limits (RPM/RPS) with automatic fallback
- **Error Resilience**: Automatic fallback to alternative models on failures
- **Structured Output**: Pydantic-validated responses with native support detection and fallback
- **Full LangChain Compatibility**: Implements BaseChatModel and Runnable interfaces
- **Usage Tracking**: Monitor requests, tokens, and success rates per model
- **Extensible**: Support for custom routing strategies and error handlers
- **Production Ready**: Comprehensive logging and error handling

## Installation

Install from PyPI:

```bash
pip install langchain-fused-model
```

For development installation:

```bash
git clone https://github.com/yourusername/langchain-fused-model
cd langchain-fused-model
pip install -e .
```

## Quick Start

Here's a simple example to get you started:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_fused_model import MultiModelManager, RoutingStrategy

# Initialize your models
models = [
    ChatOpenAI(model="gpt-4"),
    ChatOpenAI(model="gpt-3.5-turbo"),
    ChatAnthropic(model="claude-3-opus-20240229"),
]

# Create manager with priority-based routing
manager = MultiModelManager(
    models=models,
    strategy=RoutingStrategy.PRIORITY
)

# Use like any LangChain ChatModel
response = manager.invoke("What is the capital of France?")
print(response.content)
```

## Routing Strategies

The `MultiModelManager` supports multiple routing strategies to control how requests are distributed across models.

### Priority-Based Routing

Routes requests to the highest priority available model. Perfect for preferring premium models with fallback to cheaper alternatives.

```python
from langchain_fused_model import MultiModelManager, RoutingStrategy, ModelConfig

configs = [
    ModelConfig(priority=100, max_rpm=60),  # Highest priority - GPT-4
    ModelConfig(priority=50, max_rpm=120),  # Medium priority - GPT-3.5
    ModelConfig(priority=10, max_rpm=200),  # Lowest priority - Local model
]

manager = MultiModelManager(
    models=models,
    model_configs=configs,
    strategy=RoutingStrategy.PRIORITY
)
```

### Cost-Aware Routing

Automatically routes to the lowest cost model based on `cost_per_1k_tokens`. Ideal for cost optimization.

```python
configs = [
    ModelConfig(cost_per_1k_tokens=0.03),   # GPT-4 - $0.03/1k tokens
    ModelConfig(cost_per_1k_tokens=0.002),  # GPT-3.5 - $0.002/1k tokens
    ModelConfig(cost_per_1k_tokens=0.015),  # Claude - $0.015/1k tokens
]

manager = MultiModelManager(
    models=models,
    model_configs=configs,
    strategy=RoutingStrategy.COST_AWARE
)
```

### Round-Robin Routing

Distributes requests evenly across all available models. Great for load balancing.

```python
manager = MultiModelManager(
    models=models,
    strategy=RoutingStrategy.ROUND_ROBIN
)
```

### Least-Used Routing

Routes to the model with the fewest total requests. Helps balance usage across models.

```python
manager = MultiModelManager(
    models=models,
    strategy=RoutingStrategy.LEAST_USED
)
```

### Custom Strategies

You can provide a custom routing function for advanced use cases:

```python
def custom_strategy(models, configs, usage_stats, available_models):
    """Custom strategy: prefer models with highest success rate."""
    best_model = available_models[0]
    best_rate = 0.0
    
    for idx in available_models:
        stats = usage_stats.get(idx)
        if stats and stats.total_requests > 0:
            success_rate = stats.successful_requests / stats.total_requests
            if success_rate > best_rate:
                best_rate = success_rate
                best_model = idx
    
    return best_model

manager = MultiModelManager(
    models=models,
    strategy=custom_strategy
)
```

## Structured Output

Get Pydantic-validated responses from any model, with automatic fallback for models without native structured output support.

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: str = Field(description="The person's job or profession")

# Create structured output runnable
structured_manager = manager.with_structured_output(Person)

# Get validated Pydantic object
person = structured_manager.invoke("Tell me about Albert Einstein")
print(f"{person.name} was {person.age} years old and worked as a {person.occupation}")
# Output: Albert Einstein was 76 years old and worked as a Theoretical Physicist
```

The structured output handler automatically:
- Detects if the model has native structured output support
- Uses native support when available for better performance
- Falls back to prompt injection and JSON parsing when needed
- Validates all responses against your Pydantic schema

## Rate Limiting and Fallback

Configure per-model rate limits and automatic fallback behavior:

```python
from langchain_fused_model import ModelConfig

configs = [
    ModelConfig(
        priority=100,
        max_rpm=60,       # 60 requests per minute
        max_rps=2,        # 2 requests per second
        timeout=30.0,     # 30 second timeout
        retry_on_errors=[TimeoutError, ConnectionError]
    ),
    ModelConfig(
        priority=50,
        max_rpm=120,      # Fallback model with higher limits
    ),
]

manager = MultiModelManager(
    models=models,
    model_configs=configs,
    strategy=RoutingStrategy.PRIORITY,
    default_fallback=True  # Enable automatic fallback
)

# Automatically falls back if rate limit exceeded or errors occur
response = manager.invoke("Your prompt here")
```

When a model fails or hits rate limits:
1. The manager automatically selects the next available model
2. A cooldown period is set for rate-limited models
3. The request is retried with the new model
4. All failures are logged for monitoring

## LangChain Integration

The `MultiModelManager` works seamlessly with all LangChain features:

### Chains

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Use in chains with the pipe operator
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | manager | StrOutputParser()

result = chain.invoke({"topic": "programming"})
print(result)
```

### Batch Processing

```python
# Process multiple inputs in parallel
questions = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?"
]

responses = manager.batch(questions)
for response in responses:
    print(response.content)
```

### Streaming (if supported by underlying models)

```python
# Stream responses token by token
for chunk in manager.stream("Write a long story about AI"):
    print(chunk.content, end="", flush=True)
```

### Agents

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool

# Use as the LLM for agents
agent = create_openai_functions_agent(manager, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

result = agent_executor.invoke({"input": "What's the weather in Paris?"})
```

## Usage Statistics

Monitor model performance and usage:

```python
# Get statistics for all models
stats = manager._usage_tracker.get_all_stats()

for model_idx, stat in stats.items():
    print(f"\nModel {model_idx} ({models[model_idx]._llm_type}):")
    print(f"  Total requests: {stat.total_requests}")
    print(f"  Successful: {stat.successful_requests}")
    print(f"  Failed: {stat.failed_requests}")
    
    if stat.total_requests > 0:
        success_rate = stat.successful_requests / stat.total_requests
        print(f"  Success rate: {success_rate:.2%}")
    
    print(f"  Total tokens: {stat.total_tokens}")
    
    if stat.last_used:
        import datetime
        last_used = datetime.datetime.fromtimestamp(stat.last_used)
        print(f"  Last used: {last_used}")

# Get statistics for a specific model
model_0_stats = manager._usage_tracker.get_stats(0)
print(f"Model 0 has handled {model_0_stats.total_requests} requests")
```

## Advanced Configuration

### Complete Configuration Example

```python
from langchain_fused_model import MultiModelManager, ModelConfig, RoutingStrategy

configs = [
    ModelConfig(
        priority=100,              # Highest priority
        max_rpm=60,                # Rate limits
        max_rps=2,
        cost_per_1k_tokens=0.03,   # Cost tracking
        timeout=30.0,              # Request timeout
        retry_on_errors=[          # Custom retry conditions
            TimeoutError,
            ConnectionError,
        ]
    ),
    ModelConfig(
        priority=50,
        max_rpm=120,
        max_rps=5,
        cost_per_1k_tokens=0.002,
        timeout=20.0,
    ),
]

manager = MultiModelManager(
    models=models,
    model_configs=configs,
    strategy=RoutingStrategy.PRIORITY,
    default_fallback=True
)
```

### Logging Configuration

The package uses Python's standard logging module:

```python
import logging

# Enable debug logging to see model selection decisions
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('langchain_fused_model')
logger.setLevel(logging.DEBUG)

# Now you'll see detailed logs about model selection and fallback
response = manager.invoke("Test prompt")
```

## Examples

Check out the `examples/` directory for Jupyter notebooks demonstrating:

- **basic_usage.ipynb**: Getting started with MultiModelManager
- **routing_strategies.ipynb**: Comparing all routing strategies
- **structured_output.ipynb**: Working with Pydantic models and structured data

## Requirements

- Python 3.8+
- langchain-core >= 0.1.0
- pydantic >= 2.0.0

Optional dependencies for specific providers:
- langchain-openai (for OpenAI models)
- langchain-anthropic (for Anthropic models)
- langchain-google-genai (for Google models)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

To set up for development:

```bash
git clone https://github.com/yourusername/langchain-fused-model
cd langchain-fused-model
pip install -e ".[dev]"
pytest tests/
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **GitHub Issues**: https://github.com/yourusername/langchain-fused-model/issues
- **Documentation**: https://github.com/yourusername/langchain-fused-model#readme
- **Examples**: See the `examples/` directory for Jupyter notebooks

---

**Note**: This package is designed to work with any LangChain-compatible ChatModel. Make sure to install the appropriate provider packages (e.g., `langchain-openai`, `langchain-anthropic`) for the models you want to use.
