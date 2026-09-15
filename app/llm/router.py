"""Router for delegating LLM requests to the configured provider."""

import os
import logging
from app.core.config import GlobalRunConfig
from app.llm.base import BaseLLMProvider
from app.llm.provider_nvidia import NvidiaNIMProvider

logger = logging.getLogger(__name__)

class LLMRouter:
    """Factory and router for LLM providers."""

    @staticmethod
    def get_provider(config: GlobalRunConfig) -> BaseLLMProvider:
        """Instantiate the appropriate LLM provider based on config."""
        provider_name = config.llm_provider.lower()
        if provider_name == "nvidia":
            api_key = os.environ.get(config.nvidia_api_key_env, "")
            fallback_key = os.environ.get(f"{config.nvidia_api_key_env}_FALLBACK", "")
            
            keys = [k for k in [api_key, fallback_key] if k]
            
            if not keys:
                logger.warning(f"{config.nvidia_api_key_env} environment variable not set. LLM requests will fail.")
                keys = [""]
                
            return NvidiaNIMProvider(
                config=config,
                api_keys=keys,
            )
        elif provider_name == "ollama":
            raise NotImplementedError("Ollama provider not yet implemented.")
        elif provider_name == "gemini":
            raise NotImplementedError("Gemini provider not yet implemented.")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
