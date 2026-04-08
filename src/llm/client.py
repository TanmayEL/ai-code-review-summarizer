"""
LLM client - wired up to Claude via the Anthropic SDK now.

Keeping mock mode for tests and for when no API key is set,
so the pipeline still works offline. In production you'd set
ANTHROPIC_API_KEY in the environment (or .env file).

Usage:
  - LLMClient()                        -> tries Claude, falls back to mock
  - LLMClient(LLMConfig(provider="mock")) -> always mock (for tests)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    provider: str = "claude"        # "claude" or "mock"
    model: str = "claude-opus-4-6"
    max_tokens: int = 1024
    temperature: float = 0.3


class LLMClient:
    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or LLMConfig()
        self._client = None

        if self.cfg.provider == "claude":
            self._init_claude()

    def _init_claude(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set - running in mock mode. "
                "Add it to your .env file to use the real API."
            )
            self.cfg.provider = "mock"
            return

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"Claude client ready (model={self.cfg.model})")
        except ImportError:
            logger.error(
                "anthropic package not installed. "
                "Run: pip install anthropic"
            )
            self.cfg.provider = "mock"

    def generate(self, prompt: str) -> str:
        if self.cfg.provider == "mock" or self._client is None:
            snippet = prompt[:100].replace("\n", " ").strip()
            return f"[mock summary] changes related to: {snippet}..."

        import anthropic

        try:
            msg = self._client.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            raise
