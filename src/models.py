"""
src/models.py
=============
Thin, unified client wrappers for OpenAI (GPT-4o) and xAI (Grok-3).
Grok's API is OpenAI-compatible, so we reuse the openai SDK with a custom base_url.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for a single model endpoint."""
    name: str               # Human-readable label used in reports
    model_id: str           # Exact model string sent to the API
    api_key_env: str        # Environment variable holding the API key
    base_url: Optional[str] # None → use OpenAI default endpoint
    max_tokens: int = 1200
    temperature: float = 0.7


MODELS: dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        name="OpenAI GPT-4o",
        model_id="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        base_url=None,
    ),
    "grok-3": ModelConfig(
        name="xAI Grok-3",
        model_id="grok-3",
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
    ),
}


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_client(model_key: str) -> tuple[OpenAI, ModelConfig]:
    """
    Return an (OpenAI client, ModelConfig) pair for the given model key.

    Args:
        model_key: One of the keys in MODELS (e.g. "gpt-4o", "grok-3").

    Returns:
        Tuple of (client, config).

    Raises:
        KeyError: If model_key is not registered.
        ValueError: If the required API key environment variable is not set.
    """
    if model_key not in MODELS:
        raise KeyError(
            f"Unknown model '{model_key}'. Available: {list(MODELS.keys())}"
        )

    config = MODELS[model_key]
    api_key = os.getenv(config.api_key_env)

    if not api_key:
        raise ValueError(
            f"Environment variable '{config.api_key_env}' is not set. "
            f"Please add it to your .env file. See .env.example for reference."
        )

    client_kwargs: dict = {"api_key": api_key}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url

    return OpenAI(**client_kwargs), config


def chat_completion(
    client: OpenAI,
    config: ModelConfig,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Send a chat completion request and return the assistant's text content.

    Args:
        client: Configured OpenAI-compatible client.
        config: ModelConfig with model_id, max_tokens, temperature.
        system_prompt: The system-role message.
        user_prompt: The user-role message.

    Returns:
        The raw text response from the model.
    """
    response = client.chat.completions.create(
        model=config.model_id,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()
