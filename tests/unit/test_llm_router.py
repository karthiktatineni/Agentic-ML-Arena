"""Unit tests for the LLM router."""

import pytest
import os
from app.core.config import GlobalRunConfig
from app.llm.router import LLMRouter
from app.llm.provider_nvidia import NvidiaNIMProvider

def test_llm_router_nvidia():
    """Test that the router instantiates NVIDIA NIM correctly."""
    os.environ["NVIDIA_API_KEY"] = "test_key"
    if "NVIDIA_API_KEY_FALLBACK" in os.environ:
        del os.environ["NVIDIA_API_KEY_FALLBACK"]
    config = GlobalRunConfig(llm_provider="nvidia", llm_model="test-model")
    
    provider = LLMRouter.get_provider(config)
    assert isinstance(provider, NvidiaNIMProvider)
    assert provider.model == "test-model"
    assert provider.api_keys == ["test_key"]

def test_llm_router_unsupported():
    """Test that unsupported providers raise an error."""
    config = GlobalRunConfig(llm_provider="unsupported")
    with pytest.raises(ValueError, match="Unsupported LLM provider: unsupported"):
        LLMRouter.get_provider(config)

def test_llm_router_not_implemented():
    """Test that not yet implemented providers raise an error."""
    config = GlobalRunConfig(llm_provider="ollama")
    with pytest.raises(NotImplementedError):
        LLMRouter.get_provider(config)
