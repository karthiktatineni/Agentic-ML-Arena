"""Unit test and smoke test for NVIDIA NIM provider."""

import os
import pytest
from unittest.mock import MagicMock, patch
from openai import RateLimitError
from app.core.config import GlobalRunConfig
from app.llm.provider_nvidia import NvidiaNIMProvider


def test_missing_api_key_raises():
    """Verify that calling generate without API key raises descriptive ValueError."""
    config = GlobalRunConfig(nvidia_api_key_env="NON_EXISTENT_KEY_TEST")
    provider = NvidiaNIMProvider(config=config, api_keys=[])
    with pytest.raises(ValueError, match="NVIDIA API Key is missing"):
        provider.generate("Test prompt")


def test_nvidia_mocked_success():
    """Verify successful response parsing and token count extraction."""
    config = GlobalRunConfig()
    provider = NvidiaNIMProvider(config=config, api_keys=["nvapi-mock-key"])

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"status": "ok"}'
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    with patch("app.llm.provider_nvidia.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = mock_response

        response = provider.generate("Hello", json_mode=True)
        
        assert response.success is True
        assert response.content == '{"status": "ok"}'
        assert response.total_tokens == 15
        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["response_format"] == {"type": "json_object"}


def test_nvidia_rate_limit_backoff():
    """Verify that RateLimitError (429) retries and succeeds if subsequent attempt works."""
    config = GlobalRunConfig(llm_max_retries=2)
    provider = NvidiaNIMProvider(config=config, api_keys=["nvapi-mock-key"])

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Recovered after 429"
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    # Mock RateLimitError
    dummy_response = MagicMock()
    dummy_response.status_code = 429
    rate_limit_err = RateLimitError(message="429 Too Many Requests", response=dummy_response, body=None)

    with patch.object(provider, "_get_client") as mock_get_client, patch("time.sleep") as mock_sleep:
        mock_client = MagicMock()
        # Fail once with 429, then succeed on second attempt
        mock_client.chat.completions.create.side_effect = [rate_limit_err, mock_response]
        mock_get_client.return_value = mock_client

        res = provider.generate(prompt="Retry test")
        assert res.success is True
        assert res.content == "Recovered after 429"
        assert mock_sleep.call_count == 1
