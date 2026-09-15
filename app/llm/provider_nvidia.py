"""NVIDIA NIM LLM Provider using OpenAI-compatible endpoint."""

import os
import time
import random
import logging
from typing import Optional
from openai import OpenAI, RateLimitError, APIError

from app.core.config import GlobalRunConfig
from app.llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class NvidiaNIMProvider(BaseLLMProvider):
    """NVIDIA NIM hosted open-weight model provider (build.nvidia.com) with API key fallback support."""

    def __init__(self, config: Optional[GlobalRunConfig] = None, api_keys: Optional[list] = None):
        self.config = config or GlobalRunConfig()
        
        if api_keys:
            self.api_keys = api_keys
        else:
            primary = os.environ.get(self.config.nvidia_api_key_env, "")
            fallback = os.environ.get(self.config.nvidia_api_key_fallback_env, "")
            self.api_keys = [k for k in [primary, fallback] if k]
            
        self.current_key_idx = 0
        self.base_url = self.config.nvidia_base_url
        self.model = self.config.llm_model
        self.max_retries = self.config.llm_max_retries
        self.timeout = self.config.llm_request_timeout_seconds

        self._client: Optional[OpenAI] = None

    @property
    def provider_name(self) -> str:
        return "nvidia_nim"
        
    def _rotate_key(self) -> bool:
        if len(self.api_keys) > 1:
            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
            logger.info(f"Switched to fallback API key (index {self.current_key_idx}) due to rate limit/quota.")
            self._client = None  # Force re-initialization
            return True
        return False

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.api_keys:
                raise ValueError(
                    f"NVIDIA API Key is missing. Please set {self.config.nvidia_api_key_env}."
                )
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_keys[self.current_key_idx],
                timeout=self.timeout,
            )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Call NVIDIA NIM endpoint with exponential backoff on 429 rate limits."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = self._get_client()

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        attempt = 0
        backoff = 1.0

        while attempt <= self.max_retries:
            try:
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                usage = response.usage

                return LLMResponse(
                    content=content,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    model=self.model,
                    provider=self.provider_name,
                    success=True,
                )

            except RateLimitError as e:
                attempt += 1
                rotated = self._rotate_key()
                
                if attempt > self.max_retries:
                    logger.error(f"NVIDIA NIM rate limit exceeded after {self.max_retries} retries: {e}")
                    return LLMResponse(
                        content="",
                        model=self.model,
                        provider=self.provider_name,
                        success=False,
                        error=f"RateLimitError: {e}",
                    )
                
                # Exponential backoff with jitter if we didn't just rotate, or small delay if we did
                sleep_time = (backoff + random.uniform(0.1, 0.5)) if not rotated else 1.0
                logger.warning(f"Rate limited (429) on NVIDIA NIM. Retrying in {sleep_time:.2f}s (attempt {attempt}/{self.max_retries})...")
                time.sleep(sleep_time)
                if not rotated:
                    backoff *= 2.0

            except Exception as e:
                logger.error(f"NVIDIA NIM API error: {e}")
                return LLMResponse(
                    content="",
                    model=self.model,
                    provider=self.provider_name,
                    success=False,
                    error=str(e),
                )

        return LLMResponse(
            content="",
            model=self.model,
            provider=self.provider_name,
            success=False,
            error="Exhausted maximum retries.",
        )
