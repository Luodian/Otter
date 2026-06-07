"""
Configurable LLM client for benchmark evaluation.

Supports multiple LLM providers (OpenAI, MiniMax) for evaluation judging tasks
such as answer extraction and correctness scoring.

Usage:
    # Auto-detect provider from environment variables
    client = get_eval_llm_client()

    # Explicit provider selection
    client = EvalLLMClient(provider="minimax", api_key="your-key")

    # Chat completion
    content = client.chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0,
        max_tokens=256,
    )

Environment variables:
    EVAL_LLM_PROVIDER: Provider name ("openai" or "minimax")
    OPENAI_API_KEY: API key for OpenAI
    MINIMAX_API_KEY: API key for MiniMax
"""

import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import requests


PROVIDER_CONFIGS: Dict[str, Dict[str, str]] = {
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "default_model": "gpt-4-0613",
        "api_key_env": "OPENAI_API_KEY",
    },
    "minimax": {
        "api_base": "https://api.minimax.io/v1",
        # Default model: MiniMax-M3 (latest, 512K context, 128K max output, image input support).
        # Other supported models: MiniMax-M2.7, MiniMax-M2.7-highspeed.
        "default_model": "MiniMax-M3",
        "api_key_env": "MINIMAX_API_KEY",
    },
}


class EvalLLMClient:
    """Configurable LLM client for evaluation tasks.

    Supports OpenAI and MiniMax providers with automatic handling of
    provider-specific quirks (temperature clamping, think-tag stripping).
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        if provider is None:
            provider = os.environ.get("EVAL_LLM_PROVIDER", "").lower()
            if not provider:
                if os.environ.get("MINIMAX_API_KEY"):
                    provider = "minimax"
                else:
                    provider = "openai"

        self.provider = provider
        config = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS["openai"])

        self.api_base = api_base or config["api_base"]
        self.model = model or config["default_model"]
        self.api_key = api_key or os.environ.get(config["api_key_env"], "")

    def _clamp_temperature(self, temperature: float) -> float:
        """Clamp temperature for MiniMax which requires (0.0, 1.0]."""
        if self.provider == "minimax":
            return max(temperature, 0.01)
        return temperature

    def _strip_think_tags(self, content: str) -> str:
        """Strip <think>...</think> tags from MiniMax responses."""
        if self.provider == "minimax" and "<think>" in content:
            content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
        return content

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        max_tokens: int = 256,
        patience: int = 5,
        sleep_time: int = 5,
        timeout: int = 30,
    ) -> str:
        """Send a chat completion request and return the response content.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            patience: Number of retries on failure.
            sleep_time: Seconds to wait between retries.
            timeout: Request timeout in seconds.

        Returns:
            The response content string, or empty string on failure.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self._clamp_temperature(temperature),
            "max_tokens": max_tokens,
        }

        while patience > 0:
            patience -= 1
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=timeout,
                )
                response.raise_for_status()
                response_data = response.json()

                content = response_data["choices"][0]["message"]["content"].strip()
                content = self._strip_think_tags(content)
                if content:
                    return content

            except Exception as e:
                if "Rate limit" not in str(e):
                    print(e)
                time.sleep(sleep_time)

        return ""

    def chat_completion_raw(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        max_tokens: int = 256,
        timeout: int = 15,
    ) -> Tuple[str, dict]:
        """Send a chat completion request and return both content and raw response.

        Used by evaluation datasets that need the full response object
        (e.g., MMVet which tracks the model name).

        Returns:
            Tuple of (content_string, raw_response_dict).
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self._clamp_temperature(temperature),
            "max_tokens": max_tokens,
        }

        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=timeout,
        )
        response.raise_for_status()
        response_data = response.json()

        content = response_data["choices"][0]["message"]["content"].strip()
        content = self._strip_think_tags(content)
        return content, response_data


def get_eval_llm_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
) -> EvalLLMClient:
    """Factory function to create an EvalLLMClient.

    Auto-detects provider from environment variables if not specified:
    - EVAL_LLM_PROVIDER: Explicit provider name
    - MINIMAX_API_KEY: Auto-selects MiniMax if set
    - Falls back to OpenAI otherwise
    """
    return EvalLLMClient(
        provider=provider,
        api_key=api_key,
        model=model,
        api_base=api_base,
    )
