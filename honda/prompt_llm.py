# honda/prompt_llm.py
from __future__ import annotations
import json
import os
from typing import List, Dict, Any, Optional
from PIL import Image

# We try to import Qwen-VL; if unavailable, we fall back.
_QWEN_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
    _QWEN_AVAILABLE = True
except Exception:
    _QWEN_AVAILABLE = False


class LLMPrompter:
    """
    LLM-driven prompter that asks Qwen-VL to propose compact, JSON-structured
    prompt variants that better describe the reference image relative to the
    current best prompt (if any).
    Falls back to a deterministic heuristic if Qwen-VL isn't available.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        backend = (cfg.get("prompter", {}) or {}).get("backend", "qwen_vl")
        self.backend = backend
        self.last_info = {}  # emitted back to CLI as {"info": ...}

        self.qwen = None  # (tokenizer, model, processor)
        if backend == "qwen_vl" and _QWEN_AVAILABLE:
            try:
                model_id = cfg["caption"]["qwen_model"]
                self.qwen_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype="auto",
                    device_map="auto" if cfg.get("convergence", {}).get("device", "cpu") == "cuda" else None,
                )
            except Exception as e:
                self.last_info = {"llm_init_warning": f"Qwen-VL init failed: {e}; using fallback"}
                self.backend = "fallback"

        else:
            if backend == "qwen_vl":
                self.last_info = {"llm_init_warning": "transformers not present or Qwen-VL unavailable; using fallback"}
            self.backend = "fallback"

    def _prompt_system(self) -> str:
        return (
            "You are a vision prompt engineer. "
            "Given an image and the current best text prompt, produce a compact JSON with: "
            "`base` (a refined, minimal prompt), "
            "`variants` (<=7 short comma phrases to try), and `negatives` (<=5 concise bad traits). "
            "Keep language photographic, realistic, and concise. Do NOT include artist names."
        )

    def _prompt_user(self, base_prompt: str) -> str:
        guide = (
            "Return ONLY valid JSON. Example schema:\n"
            '{ "base": "subject, framing", '
            '"variants": ["studio lighting", "backlit", "golden hour", "headshot"], '
            '"negatives": ["low quality", "blurry"] }\n\n'
        )
        if base_prompt:
            return guide + f"Current best prompt:\n{base_prompt}\n"
        return guide + "Current best prompt: (none)\n"

    def _qwen_generate(self, image_path: str, base_prompt: str, max_variants: int) -> Dict[str, Any]:
        # Prepare inputs
        image = Image.open(image_path).convert("RGB")
        system = self._prompt_system()
        user = self._prompt_user(base_prompt)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user}, {"type": "image", "image": image}]},
        ]
        inputs = self.qwen_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        inputs = {k: v.to(self.qwen_model.device) for k, v in inputs.items()}

        gen_cfg = self.cfg.get("caption", {})
        max_new_tokens = int(gen_cfg.get("qwen_max_new_tokens", 96))
        temperature = float(gen_cfg.get("qwen_temperature", 0.2))
        top_p = float(gen_cfg.get("qwen_top_p", 0.9))

        with self.qwen_model.device:
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
            )

        text = self.qwen_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        # Extract last JSON block from the text
        parsed = self._extract_json(text)
        # Sanitize
        return self._sanitize(parsed, max_variants)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        # Robustly find the last {...} block
        start = text.rfind("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = text[start : end + 1]
            try:
                return json.loads(raw)
            except Exception:
                pass
        # Fallback minimal
        return {"base": "", "variants": [], "negatives": []}

    def _sanitize(self, data: Dict[str, Any], max_variants: int) -> Dict[str, Any]:
        base = str(data.get("base", "")).strip()
        negs = [str(x).strip() for x in (data.get("negatives") or []) if str(x).strip()]
        vars_ = [str(x).strip() for x in (data.get("variants") or []) if str(x).strip()]
        # Cap variants
        vars_ = vars_[: max(1, int(max_variants))]
        # De-duplicate terse phrases
        def dedupe(seq):
            seen = set()
            out = []
            for s in seq:
                if s not in seen:
                    out.append(s)
                    seen.add(s)
            return out
        return {"base": base, "variants": dedupe(vars_), "negatives": dedupe(negs)}

    def _fallback(self, image_path: str, base_prompt: str, max_variants: int) -> Dict[str, Any]:
        """
        Deterministic, dependency-light fallback:
        - Keep BLIP caption or supplied base_prompt as base
        - Offer a small, balanced set of neutral photographic modifiers
        """
        default_vars = [
            "in a photograph",
            "soft light",
            "studio lighting",
            "golden hour",
            "backlit",
            "headshot",
            "shallow depth of field",
        ]
        negs = ["low quality", "blurry", "deformed", "oversaturated"]
        return {
            "base": base_prompt,
            "variants": default_vars[: max(1, int(max_variants))],
            "negatives": negs,
        }

    def propose(self, image_path: str, base_prompt: str, max_variants: int) -> Dict[str, Any]:
        if self.backend == "qwen_vl" and _QWEN_AVAILABLE and self.qwen_model is not None:
            try:
                return self._qwen_generate(image_path, base_prompt, max_variants)
            except Exception as e:
                self.last_info = {"llm_error": f"{e}; using fallback"}
                return self._fallback(image_path, base_prompt, max_variants)
        return self._fallback(image_path, base_prompt, max_variants)
