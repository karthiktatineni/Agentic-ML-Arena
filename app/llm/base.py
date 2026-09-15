"""Base interface for pluggable LLM providers."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Structured response from an LLM call with token consumption metadata."""
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    provider: str = ""
    success: bool = True
    error: Optional[str] = None
    cached: bool = False


class BaseLLMProvider(ABC):
    """Abstract class for all LLM providers (NVIDIA NIM, Gemini, Ollama)."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Execute a text generation call."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier."""
        pass
