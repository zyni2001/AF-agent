"""Shared LLM provider configuration for Gemini Developer API and Vertex AI."""

import os
from typing import Dict, List, Optional


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def use_vertex_ai() -> bool:
    """Return True when requests should be routed through Vertex AI."""
    return _is_truthy(os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"))


def get_vertex_project() -> Optional[str]:
    return os.environ.get("VERTEXAI_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")


def get_vertex_location() -> str:
    return os.environ.get("VERTEXAI_LOCATION") or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"


def get_gemini_model() -> str:
    if use_vertex_ai():
        return os.environ.get("VERTEXAI_GEMINI_MODEL", "vertex_ai/gemini-2.5-flash")
    return os.environ.get("GEMINI_MODEL", "gemini/gemini-2.5-flash")


def get_api_keys_from_env() -> List[str]:
    """Load Gemini Developer API keys from GEMINI_API_KEY."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    return [key.strip() for key in api_key.split(",") if key.strip()]


def build_litellm_kwargs(messages: List[Dict[str, str]], temperature: float = 0.0) -> Dict[str, object]:
    """Build LiteLLM kwargs for the configured provider."""
    kwargs: Dict[str, object] = {
        "messages": messages,
        "model": get_gemini_model(),
        "temperature": temperature,
    }

    if use_vertex_ai():
        project = get_vertex_project()
        if not project:
            raise ValueError(
                "Vertex AI is enabled but neither VERTEXAI_PROJECT nor GOOGLE_CLOUD_PROJECT is set"
            )
        kwargs["vertex_project"] = project
        kwargs["vertex_location"] = get_vertex_location()

    return kwargs
