# honda/prompt_blocks.py
from __future__ import annotations
from typing import List

class PromptManager:
    """
    Carries the winning prompt forward, optionally keeping a small number of
    trailing comma-separated modifiers between steps for stability.
    """
    def __init__(self, initial_caption: str, carry_forward_modifiers: int = 1):
        self.base = initial_caption.strip()
        self.carry = max(0, int(carry_forward_modifiers))

    def expand_with_variants(self, base: str, variants: List[str]) -> List[str]:
        # Always include the bare base; then base + ", variant" for each variant
        out = [base]
        for v in variants:
            v = v.strip().strip(",")
            if not v:
                continue
            out.append(f"{base}, {v}")
        return out

    def refine_with_best(self, best_prompt: str) -> str:
        """
        Keep up to N trailing modifiers (comma-separated phrases) from the best prompt.
        """
        if self.carry <= 0:
            self.base = best_prompt.strip()
            return self.base

        parts = [p.strip() for p in best_prompt.split(",") if p.strip()]
        if len(parts) <= 1:
            self.base = best_prompt.strip()
            return self.base

        head = parts[0]
        tail = parts[1:][-self.carry:]  # keep last N
        new_base = ", ".join([head] + tail)
        self.base = new_base
        return self.base
