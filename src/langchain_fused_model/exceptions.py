"""Custom exceptions for langchain-fused-model.

This module defines custom exception classes for handling various error
conditions in the MultiModelManager system.
"""


class MultiModelError(Exception):
    """Base exception for all multi-model related errors.
    
    This is the parent class for all custom exceptions in the
    langchain-fused-model package. Catch this to handle all
    package-specific errors.
    """
    pass


class AllModelsFailedError(MultiModelError):
    """Raised when all configured models fail to respond.
    
    This exception is raised when the MultiModelManager has exhausted
    all available models and none were able to successfully complete
    the request. The exception includes details about each model's
    failure for debugging purposes.
    
    Attributes:
        errors: Dictionary mapping model indices to error details,
            including model type, error type, and error message.
    
    Example:
        >>> try:
        ...     response = manager.invoke("prompt")
        ... except AllModelsFailedError as e:
        ...     print(f"All models failed: {e.errors}")
    """
    
    def __init__(self, errors: dict, message: str = "All models failed to respond"):
        """Initialize with error details from all models.
        
        Args:
            errors: Dictionary mapping model index to error details.
                Each entry should contain 'model_type', 'error_type',
                and 'error_message' keys.
            message: Custom error message describing the failure.
        """
        self.errors = errors
        super().__init__(message)


class RateLimitExceededError(MultiModelError):
    """Raised when all models are currently rate limited.
    
    This exception is raised when all configured models have reached
    their rate limits (RPM/RPS) or are in cooldown periods, and no
    models are available to handle the request.
    
    Example:
        >>> try:
        ...     response = manager.invoke("prompt")
        ... except RateLimitExceededError:
        ...     print("All models are rate limited, try again later")
    """
    
    def __init__(self, message: str = "All models are currently rate limited"):
        """Initialize with rate limit message.
        
        Args:
            message: Custom error message describing the rate limit condition.
        """
        super().__init__(message)


class StructuredOutputError(MultiModelError):
    """Raised when structured output parsing or validation fails.
    
    This exception is raised when the system fails to parse a model's
    response into the requested Pydantic schema. This can occur due to:
    - Invalid JSON in the response
    - JSON that doesn't match the schema
    - Pydantic validation errors
    
    Attributes:
        original_error: The underlying exception that caused the failure
            (e.g., JSONDecodeError, ValidationError).
    
    Example:
        >>> try:
        ...     result = structured_manager.invoke("prompt")
        ... except StructuredOutputError as e:
        ...     print(f"Failed to parse: {e}")
        ...     print(f"Original error: {e.original_error}")
    """
    
    def __init__(self, message: str, original_error: Exception = None):
        """Initialize with parsing error details.
        
        Args:
            message: Description of the parsing error.
            original_error: The original exception that caused the failure,
                if available.
        """
        self.original_error = original_error
        super().__init__(message)
