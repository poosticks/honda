# honda/prompter_openai.py
from __future__ import annotations
import json
from typing import Any, Dict, List

from .openai_compat import OpenAICompat


class PromptVariantGenerator:
    """
    Generates *feedback-aware* prompt variants via an OpenAI-compatible LLM (LM Studio / Qwen).
    It preserves ADD:/STYLE: tokens when present, so feedback is enforced in the next round.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        ocfg = self.cfg.get("openai", {})
        self.client = OpenAICompat(
            api_base=ocfg["api_base"],
            model=ocfg["model"],
            request_timeout_s=int(ocfg.get("request_timeout_s", 60)),
            temperature=float(ocfg.get("temperature", 0.2)),
            top_p=float(ocfg.get("top_p", 0.9)),
            max_tokens=int(ocfg.get("max_tokens", 256)),
        )

    @staticmethod
    def _safe_json_list(txt: str) -> List[str]:
        s = txt.strip()
        if "```" in s:
            parts = s.split("```")
            if len(parts) >= 2:
                s = parts[1]
                if s.lower().startswith("json"):
                    s = s[4:].strip()
        try:
            arr = json.loads(s)
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return [x.strip() for x in arr if x.strip()]
        except Exception:
            pass
        # fallback: split lines
        return [ln.strip("- •").strip() for ln in s.splitlines() if ln.strip()]

    @staticmethod
    def _apply_feedback(base: str, fb: Dict[str, Any] | None) -> str:
        if not fb:
            return base
        add = ", ".join(fb.get("add", []))
        sty = ", ".join(fb.get("style", []))
        pieces = [base]
        if add:
            pieces.append(f"ADD: {add}")
        if sty:
            pieces.append(f"STYLE: {sty}")
        # Keep short and structured for SD
        return " | ".join(pieces)

    def generate_variants(
        self,
        base_prompt: str,
        max_variants: int = 7,
        feedback: Dict[str, Any] | None = None,
    ) -> List[str]:
        base_with_fb = self._apply_feedback(base_prompt, feedback)
        sys = (
            "You produce concise Stable Diffusion prompt variants. "
            "If the prompt includes 'ADD:' or 'STYLE:' sections, you MUST preserve those tokens verbatim. "
            "Do not contradict them. Output a JSON array of strings."
        )
        usr = (
            f"Base prompt:\n{base_with_fb}\n\n"
            "Rules:\n"
            "- Keep key subject tokens (identity/scene).\n"
            "- Preserve and incorporate tokens in 'ADD:' and 'STYLE:' sections (you may augment, not delete).\n"
            "- ≤ 30 words each. No camera jargon unless STYLE asks for it.\n"
            f"- Return ≤ {max_variants} variants as a JSON list of strings."
        )
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": usr},
        ]
        out = self.client.chat(messages)
        return self._safe_json_list(out)
