"""LLM-based simulator — requires extra [llm].

Do NOT import this module unless the [llm] extra is installed.
"""

from __future__ import annotations

from typing import Any

from deeprl_recsys.core.logging import get_logger
from deeprl_recsys.environment.simulators.base_sim import BaseSimulator

logger = get_logger(__name__)

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except (ImportError, OSError):
    HAS_TRANSFORMERS = False


class LLMSimulator(BaseSimulator):
    """Simulates user responses using an LLM.

    Requires: ``pip install deeprl-recsys[llm]``
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: int = -1,
        max_new_tokens: int = 20,
        **kwargs: Any,
    ) -> None:
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "LLMSimulator requires the [llm] extra. "
                "Install with: pip install deeprl-recsys[llm]"
            )
        self._config = kwargs
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        logger.info("llm_sim_init", model=model_name, device=device)
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            pad_token_id=50256,  # common EOS token ID for GPT-2
        )

    def simulate_response(self, user: dict[str, Any], item: int) -> float:
        """Simulate a click/reward given the user context and recommended item.
        
        Prompts the LLM and parses the response to extract a reward signal.
        """
        user_id = user.get("user_id", "Unknown")
        prompt = (
            f"User {user_id} was recommended item {item}. "
            f"Does the user like this item? Answer YES or NO.\nAnswer:"
        )

        try:
            results = self.generator(
                prompt, max_new_tokens=self.max_new_tokens, num_return_sequences=1
            )
            generated_text = results[0]["generated_text"]
            
            # Simple parsing: check if "YES" appears after the prompt
            response_only = generated_text[len(prompt):].strip().upper()
            if "YES" in response_only:
                return 1.0
            elif "NO" in response_only:
                return 0.0
            
            # Fallback to a neutral/heuristic signal for unexpected responses
            return 0.5
        except Exception as exc:
            logger.warning("llm_sim_error", error=str(exc))
            return 0.0
