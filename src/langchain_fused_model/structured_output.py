"""Structured output handling for MultiModelManager."""

import json
import logging
import re
from typing import Type, List, Any, Dict
from pydantic import BaseModel, ValidationError
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable

from .exceptions import StructuredOutputError

# Set up logger
logger = logging.getLogger(__name__)


class StructuredOutputHandler:
    """Manages structured output for models with native support detection and fallback.
    
    This class provides methods to handle structured output from language models,
    automatically detecting whether a model has native structured output support
    and falling back to prompt-based JSON extraction when needed.
    """
    
    def __init__(self):
        """Initialize the StructuredOutputHandler."""
        pass
    
    def has_native_support(self, model: Any) -> bool:
        """Check if model has native structured output support.
        
        Detects whether the model has a with_structured_output method,
        which indicates native support for structured output.
        
        Args:
            model: The ChatModel instance to check.
        
        Returns:
            True if the model has native structured output support, False otherwise.
        """
        has_support = hasattr(model, 'with_structured_output') and callable(
            getattr(model, 'with_structured_output')
        )
        
        logger.debug(
            f"Model {model._llm_type if hasattr(model, '_llm_type') else 'unknown'} "
            f"has native structured output support: {has_support}"
        )
        
        return has_support
    
    def invoke_native(
        self,
        model: Any,
        schema: Type[BaseModel],
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> BaseModel:
        """Invoke model with native structured output.
        
        Delegates to the model's native with_structured_output method.
        
        Args:
            model: The ChatModel instance with native support.
            schema: Pydantic model class defining the expected output structure.
            messages: List of messages to send to the model.
            **kwargs: Additional keyword arguments to pass to the model.
        
        Returns:
            Instance of the Pydantic model with parsed data.
        
        Raises:
            StructuredOutputError: If the native method fails.
        """
        try:
            logger.debug(f"Invoking native structured output for schema {schema.__name__}")
            
            # Create a structured output runnable
            structured_model = model.with_structured_output(schema, **kwargs)
            
            # Invoke the structured model
            result = structured_model.invoke(messages)
            
            logger.debug(f"Successfully parsed native structured output")
            
            return result
            
        except Exception as e:
            logger.error(f"Native structured output failed: {type(e).__name__}: {str(e)}")
            raise StructuredOutputError(
                f"Native structured output failed: {str(e)}",
                original_error=e
            )

    def invoke_with_parsing(
        self,
        model: Any,
        schema: Type[BaseModel],
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> BaseModel:
        """Invoke model and parse output to structured format.
        
        Uses prompt injection to request JSON output, then extracts and
        validates the JSON against the Pydantic schema.
        
        Args:
            model: The ChatModel instance without native support.
            schema: Pydantic model class defining the expected output structure.
            messages: List of messages to send to the model.
            **kwargs: Additional keyword arguments to pass to the model.
        
        Returns:
            Instance of the Pydantic model with parsed data.
        
        Raises:
            StructuredOutputError: If JSON extraction or validation fails.
        """
        try:
            logger.debug(f"Invoking with parsing fallback for schema {schema.__name__}")
            
            # Inject format instructions into messages
            modified_messages = self._inject_format_instructions(messages, schema)
            
            # Invoke the model with modified messages
            result = model.invoke(modified_messages, **kwargs)
            
            # Extract text from result
            if hasattr(result, 'content'):
                response_text = result.content
            elif isinstance(result, str):
                response_text = result
            else:
                response_text = str(result)
            
            logger.debug(f"Received response, extracting JSON")
            
            # Extract JSON from response
            json_str = self._extract_json(response_text)
            
            # Parse JSON to Pydantic model
            parsed_model = self._parse_to_model(json_str, schema)
            
            logger.debug(f"Successfully parsed structured output with fallback")
            
            return parsed_model
            
        except StructuredOutputError:
            # Re-raise StructuredOutputError as-is
            raise
        except Exception as e:
            logger.error(f"Structured output parsing failed: {type(e).__name__}: {str(e)}")
            raise StructuredOutputError(
                f"Failed to parse structured output: {str(e)}",
                original_error=e
            )
    
    def _inject_format_instructions(
        self,
        messages: List[BaseMessage],
        schema: Type[BaseModel]
    ) -> List[BaseMessage]:
        """Add JSON schema instructions to messages.
        
        Injects format instructions into the message list to guide the model
        to produce JSON output matching the Pydantic schema.
        
        Args:
            messages: Original list of messages.
            schema: Pydantic model class defining the expected output structure.
        
        Returns:
            Modified list of messages with format instructions added.
        """
        # Create JSON output parser to get format instructions
        parser = JsonOutputParser(pydantic_object=schema)
        format_instructions = parser.get_format_instructions()
        
        # Create instruction message
        instruction_text = (
            f"You must respond with valid JSON that matches the following schema. "
            f"Do not include any text before or after the JSON object.\n\n"
            f"{format_instructions}"
        )
        
        # Add instruction as a system message at the beginning
        modified_messages = [SystemMessage(content=instruction_text)] + list(messages)
        
        logger.debug(f"Injected format instructions for schema {schema.__name__}")
        
        return modified_messages
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from model response using regex.
        
        Attempts to extract a JSON object or array from the response text,
        handling cases where the model includes additional text.
        
        Args:
            text: The raw response text from the model.
        
        Returns:
            Extracted JSON string.
        
        Raises:
            StructuredOutputError: If no valid JSON is found in the response.
        """
        # Try to find JSON object (starts with { and ends with })
        json_object_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        object_matches = re.findall(json_object_pattern, text, re.DOTALL)
        
        if object_matches:
            # Return the last match (most likely to be complete)
            json_str = object_matches[-1]
            logger.debug(f"Extracted JSON object from response")
            return json_str
        
        # Try to find JSON array (starts with [ and ends with ])
        json_array_pattern = r'\[(?:[^\[\]]|(?:\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]))*\]'
        array_matches = re.findall(json_array_pattern, text, re.DOTALL)
        
        if array_matches:
            # Return the last match (most likely to be complete)
            json_str = array_matches[-1]
            logger.debug(f"Extracted JSON array from response")
            return json_str
        
        # If no JSON found, try to use the entire text (might be valid JSON)
        text = text.strip()
        if text.startswith('{') or text.startswith('['):
            logger.debug(f"Using entire response as JSON")
            return text
        
        # No JSON found
        logger.error(f"No valid JSON found in response")
        raise StructuredOutputError(
            f"Could not extract valid JSON from model response. Response: {text[:200]}..."
        )
    
    def _parse_to_model(self, json_str: str, schema: Type[BaseModel]) -> BaseModel:
        """Parse JSON string and validate with Pydantic.
        
        Parses the JSON string and validates it against the Pydantic schema.
        
        Args:
            json_str: JSON string to parse.
            schema: Pydantic model class for validation.
        
        Returns:
            Instance of the Pydantic model with validated data.
        
        Raises:
            StructuredOutputError: If JSON parsing or validation fails.
        """
        try:
            # Parse JSON string to dictionary
            data = json.loads(json_str)
            logger.debug(f"Successfully parsed JSON string")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            raise StructuredOutputError(
                f"Failed to decode JSON: {str(e)}. JSON string: {json_str[:200]}...",
                original_error=e
            )
        
        try:
            # Validate and create Pydantic model instance
            model_instance = schema(**data) if isinstance(data, dict) else schema.parse_obj(data)
            logger.debug(f"Successfully validated data against schema {schema.__name__}")
            
            return model_instance
            
        except ValidationError as e:
            logger.error(f"Pydantic validation error: {str(e)}")
            raise StructuredOutputError(
                f"Failed to validate data against schema {schema.__name__}: {str(e)}",
                original_error=e
            )
        except Exception as e:
            logger.error(f"Unexpected error during model creation: {str(e)}")
            raise StructuredOutputError(
                f"Failed to create model instance: {str(e)}",
                original_error=e
            )
