"""
src/generator.py
================
Email generation logic using the advanced prompt template.

Technique: Chain-of-Thought (CoT) + Few-Shot Examples + Role-Playing
  - The system prompt assigns an expert persona (Role-Playing).
  - The user prompt contains 2 worked examples showing reasoning + output (Few-Shot).
  - Each example and the live task use explicit [STEP 1-5] reasoning (CoT).
  - The model reasons before writing — this maximises fact inclusion and tone accuracy.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from src.models import ModelConfig, OpenAI, chat_completion

# Path to the master prompt template
_TEMPLATE_PATH = Path(__file__).parent.parent / "prompts" / "email_prompt_template.txt"


def _load_template() -> str:
    """Load and return the raw prompt template text."""
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def _build_prompts(intent: str, key_facts: list[str], tone: str) -> tuple[str, str]:
    """
    Split the master template into system_prompt and user_prompt,
    then inject the live task variables into the user_prompt.

    The template file is split at the LIVE TASK TEMPLATE section marker.
    Everything above that marker becomes the system prompt (persona + examples).
    Everything in/after that section is the user prompt with variables filled in.

    Args:
        intent: Core purpose of the email.
        key_facts: List of fact strings that must appear in the email.
        tone: Desired communication style.

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    raw = _load_template()

    # Split at the LIVE TASK TEMPLATE section
    split_marker = "LIVE TASK TEMPLATE"
    parts = raw.split(split_marker, maxsplit=1)

    system_prompt_section = parts[0].strip()
    live_task_section = parts[1].strip() if len(parts) > 1 else ""

    # Format key_facts as a bulleted list
    key_facts_formatted = "\n".join(f"    - {fact}" for fact in key_facts)

    # Inject variables into the live task section
    user_prompt = live_task_section.format(
        intent=intent,
        key_facts_formatted=key_facts_formatted,
        tone=tone,
    )

    return system_prompt_section, user_prompt


def extract_email_body(raw_response: str) -> str:
    """
    Extract the content inside <EMAIL>...</EMAIL> tags from the model's response.
    If the tags are not found, return the full response as a fallback.

    Args:
        raw_response: Raw text returned by the model.

    Returns:
        The extracted email text (subject line + body).
    """
    match = re.search(r"<EMAIL>(.*?)</EMAIL>", raw_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: return full response if tags are missing
    return raw_response.strip()


def generate_email(
    client: OpenAI,
    config: ModelConfig,
    intent: str,
    key_facts: list[str],
    tone: str,
    return_raw: bool = False,
) -> str:
    """
    Generate a professional email for the given scenario.

    Args:
        client: Configured LLM client.
        config: ModelConfig (model_id, temperature, etc.).
        intent: Core purpose of the email.
        key_facts: List of facts that must appear in the generated email.
        tone: Desired communication tone.
        return_raw: If True, return the full model response including CoT reasoning.
                    If False (default), return only the extracted email body.

    Returns:
        The generated email string (or full response if return_raw=True).
    """
    system_prompt, user_prompt = _build_prompts(intent, key_facts, tone)
    raw_response = chat_completion(client, config, system_prompt, user_prompt)

    if return_raw:
        return raw_response
    return extract_email_body(raw_response)
