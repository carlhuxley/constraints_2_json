"""
LLM Interpreter - converts business rules to JSON Schema constraints.

This module provides functionality to use an LLM to interpret free-text
business rules and convert them to structured JSON Schema constraints.
"""
import json
import os
import re
from typing import Optional, Protocol

import requests


class LLMClient(Protocol):
    """Protocol defining the interface for LLM clients."""

    def complete(self, prompt: str) -> str:
        """Send prompt to LLM and return response text."""
        ...


class OpenRouterClient:
    """LLM client for OpenRouter API."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize OpenRouter client.

        Args:
            model: Model identifier (e.g., "deepseek/deepseek-chat")
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def complete(self, prompt: str) -> str:
        """
        Send prompt to OpenRouter and return response text.

        Args:
            prompt: Prompt to send to the LLM

        Returns:
            Response text from the LLM
        """
        if not self.api_key:
            print("Warning: OPENROUTER_API_KEY not set.")
            return ""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/google/gemini-cli",  # Optional
            "X-Title": "Gemini CLI",  # Optional
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,  # Stable output for structured data
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(data),
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling OpenRouter: {e}")
            return ""


def interpret_business_rule(
    field_name: str,
    field_type: str,
    business_rule: str,
    llm_client: LLMClient
) -> dict:
    """
    Convert business rule text to JSON Schema constraints.

    Uses an LLM to interpret the business rule and extract
    structured constraints that can be applied to a JSON Schema.

    Args:
        field_name: Name of the field being constrained
        field_type: JSON Schema type of the field (string, integer, etc.)
        business_rule: Free-text business rule to interpret
        llm_client: Client for making LLM API calls

    Returns:
        Dict of JSON Schema constraints derived from the rule
    """
    prompt = _build_prompt(field_name, field_type, business_rule)

    try:
        response = llm_client.complete(prompt)
        return _parse_response(response)
    except Exception:
        return {}


def _build_prompt(field_name: str, field_type: str, business_rule: str) -> str:
    """Build the prompt for the LLM."""
    return f"""Convert this business rule to JSON Schema constraints.

Field: {field_name}
Type: {field_type}
Rule: {business_rule}

Return ONLY valid JSON with applicable constraints from this list:
- minimum, maximum (for numbers)
- minLength, maxLength (for strings)
- pattern (regex for strings)
- enum (array of allowed values)
- format (standard formats like email, uri, date)
- exclusiveMinimum, exclusiveMaximum (boolean flags)

Example output: {{"minimum": 18, "maximum": 120}}

JSON response:"""


def _parse_response(response: str) -> dict:
    """
    Parse LLM response to extract JSON constraints.

    Handles various response formats including JSON embedded in text.

    Args:
        response: Raw response from LLM

    Returns:
        Parsed constraints dict, empty dict if parsing fails
    """
    if not response:
        return {}

    response = response.strip()

    # Try direct JSON parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from text (handle "Here is the result: {...}")
    json_match = re.search(r'\{[^{}]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try to find JSON with nested structures
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {}
