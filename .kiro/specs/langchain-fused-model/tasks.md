# Implementation Plan

- [x] 1. Set up project structure and packaging configuration





  - Create src/langchain_fused_model/ directory with __init__.py
  - Create pyproject.toml with package metadata, dependencies (langchain, pydantic), and build configuration
  - Create LICENSE file (MIT or Apache 2.0)
  - Create basic README.md with project description
  - Create .gitignore for Python projects
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 10.5_

- [x] 2. Implement core data models and exceptions





  - Create exceptions.py with MultiModelError, AllModelsFailedError, RateLimitExceededError, StructuredOutputError classes
  - Create ModelConfig dataclass in manager.py with priority, max_rpm, max_rps, cost_per_1k_tokens, timeout, retry_on_errors fields
  - Create UsageStats dataclass in usage_tracker.py with total_requests, successful_requests, failed_requests, total_tokens, last_used fields
  - _Requirements: 2.2, 4.2, 4.3, 5.3_

- [x] 3. Implement UsageTracker class





  - Create usage_tracker.py with UsageTracker class
  - Implement __init__ method to initialize stats dictionary
  - Implement record_request method to update statistics (success/failure, tokens, timestamp)
  - Implement get_stats method to retrieve statistics for a specific model
  - Implement get_all_stats method to retrieve all model statistics
  - _Requirements: 3.3, 4.4, 5.4_

- [x] 4. Implement RateLimiter class





  - Create rate_limiter.py with RateLimiter class
  - Implement __init__ method to initialize request_times and cooldowns dictionaries
  - Implement is_available method to check if model is not rate limited or in cooldown
  - Implement record_request method to store request timestamp
  - Implement set_cooldown method to set temporary cooldown for a model
  - Implement _check_rpm method to verify requests per minute limit
  - Implement _check_rps method to verify requests per second limit
  - Use deque with time-window tracking for efficient rate limit checking
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 5. Implement routing strategies





  - Create strategy.py with RoutingStrategy enum (PRIORITY, ROUND_ROBIN, LEAST_USED, COST_AWARE)
  - Create StrategySelector class with select method
  - Implement _priority_select method to choose highest priority available model
  - Implement _round_robin_select method to distribute requests evenly across models
  - Implement _least_used_select method to select model with lowest usage count
  - Implement _cost_aware_select method to prioritize lowest cost models
  - Support custom callable strategies for extensibility
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 6. Implement MultiModelManager core functionality





  - Create manager.py with MultiModelManager class inheriting from BaseChatModel
  - Implement __init__ method accepting models list, model_configs, strategy, and default_fallback parameters
  - Initialize UsageTracker, RateLimiter, and StrategySelector instances
  - Implement _llm_type property returning "multi-model-manager"
  - Implement _identifying_params property for model identification
  - _Requirements: 2.1, 2.2, 2.3, 2.5_

- [x] 7. Implement model selection and fallback logic




  - Implement _select_model method in MultiModelManager using StrategySelector and RateLimiter
  - Filter available models based on rate limits and cooldowns
  - Return selected model and its index
  - Implement _invoke_with_fallback method to attempt request with automatic fallback
  - Handle rate limit exceptions, timeout exceptions, and other configured errors
  - Log model selection and fallback events
  - Raise AllModelsFailedError when all models are exhausted
  - _Requirements: 4.1, 5.1, 5.2, 5.3, 5.4_

- [x] 8. Implement _generate method for LangChain integration





  - Implement _generate method in MultiModelManager accepting messages, stop, run_manager, and kwargs
  - Emit on_llm_start callback if run_manager provided
  - Call _select_model to choose initial model
  - Call _invoke_with_fallback to execute request with fallback support
  - Record usage statistics via UsageTracker
  - Emit on_llm_end callback on success or on_llm_error on failure
  - Return ChatResult with generation results
  - _Requirements: 2.4, 5.5, 7.1, 7.4_
-

- [x] 9. Implement structured output handling




  - Create structured_output.py with StructuredOutputHandler class
  - Implement has_native_support method to detect if model has with_structured_output method
  - Implement invoke_native method to delegate to model's native structured output
  - Implement invoke_with_parsing method for fallback structured output
  - Implement _inject_format_instructions method to add JSON schema instructions to messages using JsonOutputParser
  - Implement _extract_json method using regex to extract JSON from response text
  - Implement _parse_to_model method to parse JSON string and validate with Pydantic
  - Handle JSON decode errors and validation errors with StructuredOutputError
  - _Requirements: 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 10. Implement with_structured_output method










  - Implement with_structured_output method in MultiModelManager accepting schema (Pydantic class) and kwargs
  - Create StructuredOutputHandler instance
  - Return a Runnable that wraps the manager and applies structured output logic
  - Check native support for each selected model and use appropriate method
  - _Requirements: 6.1, 6.2, 6.3_

- [x] 11. Implement Runnable interface methods





  - Verify invoke method works correctly (inherited from BaseChatModel)
  - Verify batch method works correctly (inherited from BaseChatModel)
  - Verify pipe method works correctly for chaining with other Runnables
  - Test integration with LangChain chains and agents
  - _Requirements: 7.1, 7.2, 7.3, 7.5_

- [x] 12. Add logging throughout the system





  - Add DEBUG level logs for model selection decisions in _select_model
  - Add INFO level logs for rate limit events in RateLimiter
  - Add WARNING level logs for fallback events in _invoke_with_fallback
  - Add ERROR level logs for model failures in _invoke_with_fallback
  - Include model identifier in all log messages
  - _Requirements: 4.5, 5.4_

- [x] 13. Create test infrastructure





  - Create tests/ directory with __init__.py
  - Create conftest.py with pytest fixtures for dummy models
  - Create dummy_chat_model fixture that returns mock ChatModel
  - Create rate_limited_model fixture that raises rate limit errors
  - Create timeout_model fixture that raises timeout errors
  - Create structured_output_model fixture with native structured output support
  - _Requirements: 9.5_

- [ ]* 14. Write unit tests for RateLimiter
  - Create test_rate_limiter.py
  - Write test for RPM limit enforcement
  - Write test for RPS limit enforcement
  - Write test for cooldown behavior
  - Write test for availability checking with multiple models
  - _Requirements: 9.1_

- [ ]* 15. Write unit tests for routing strategies
  - Create test_strategies.py
  - Write test for PRIORITY strategy selecting highest priority model
  - Write test for ROUND_ROBIN strategy distributing requests evenly
  - Write test for LEAST_USED strategy selecting least utilized model
  - Write test for COST_AWARE strategy selecting lowest cost model
  - Write test for custom strategy function
  - _Requirements: 9.4_

- [ ]* 16. Write unit tests for structured output
  - Create test_structured_output.py
  - Write test for native structured output detection and delegation
  - Write test for fallback prompt injection with format instructions
  - Write test for JSON extraction from model responses
  - Write test for Pydantic validation of parsed JSON
  - Write test for error handling with invalid JSON
  - Write test for error handling with validation failures
  - _Requirements: 9.2, 9.3_

- [ ]* 17. Write unit tests for MultiModelManager
  - Create test_manager.py
  - Write test for initialization with various configurations
  - Write test for model selection with different strategies
  - Write test for automatic fallback on rate limit errors
  - Write test for automatic fallback on timeout errors
  - Write test for AllModelsFailedError when all models fail
  - Write test for LangChain invoke method
  - Write test for LangChain batch method
  - Write test for piping with other Runnables
  - Write test for callback emission (on_llm_start, on_llm_end, on_llm_error)
  - _Requirements: 9.1, 9.4_

- [x] 18. Create documentation and examples





  - Update README.md with installation instructions (pip install langchain-fused-model)
  - Add usage examples to README.md showing basic MultiModelManager setup
  - Add routing strategy configuration examples to README.md
  - Create examples/ directory
  - Create basic_usage.ipynb demonstrating simple model management
  - Create routing_strategies.ipynb showing all routing strategies
  - Create structured_output.ipynb demonstrating structured output with Pydantic
  - Add docstrings to all public classes and methods
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 19. Create contribution guidelines and CI/CD





  - Create CONTRIBUTING.md with contribution guidelines and development setup
  - Create .github/workflows/test.yml for automated testing with pytest
  - Create .github/workflows/lint.yml for code linting with ruff or flake8
  - Configure GitHub repository with issues enabled
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 20. Prepare for PyPI distribution
  - Verify pyproject.toml has correct package metadata (name, version, description, authors, license)
  - Verify all dependencies are properly specified with version constraints
  - Test package build with `python -m build`
  - Test package installation in clean virtual environment
  - Create release documentation with version notes
  - _Requirements: 1.1, 1.5_
