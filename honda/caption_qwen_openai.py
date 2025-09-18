# honda/caption_qwen_openai.py
from __future__ import annotations
import base64, json, urllib.request, urllib.error, pathlib

class ImageCaptionerQwenOpenAI:
    """
    Uses an OpenAI-compatible /chat/completions endpoint (LM Studio) with a *vision* model (Qwen2.5-VL).
    The image is sent as a data URL so no external hosting is required.

    Config expects:
      cfg["caption"]["openai"] = {
        "api_base": ".../v1",
        "model": "qwen2.5-vl-3b-instruct",
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 128,
        "request_timeout_s": 60,
      }
    """

    def __init__(self, cfg: dict):
        self.api_base = cfg["caption"]["openai"]["api_base"].rstrip("/")
        self.model = cfg["caption"]["openai"]["model"]
        self.temperature = float(cfg["caption"]["openai"].get("temperature", 0.2))
        self.top_p = float(cfg["caption"]["openai"].get("top_p", 0.9))
        self.max_tokens = int(cfg["caption"]["openai"].get("max_tokens", 128))
        self.timeout = int(cfg["caption"]["openai"].get("request_timeout_s", 60))

        # Instruction keeps output short & caption-focused
        self.system_msg = (
            "You are an image captioner. Describe the salient subject(s), "
            "style and setting in one short sentence suitable as a Stable Diffusion prompt. "
            "Avoid negatives and camera metadata."
        )

    @staticmethod
    def _image_to_data_url(img_path: str) -> str:
        p = pathlib.Path(img_path)
        ext = p.suffix.lower()
        mime = "image/jpeg"
        if ext in (".png",):
            mime = "image/png"
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def caption(self, image_path: str) -> str:
        data_url = self._image_to_data_url(image_path)
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image as a single-sentence SD prompt."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }
        req = urllib.request.Request(
            self.api_base + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip()
            # Strip surrounding quotes if LLM adds them
            if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
                text = text[1:-1]
            return text
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Qwen OpenAI caption HTTPError: {e.code} {e.reason}")
        except Exception as e:
            raise RuntimeError(f"Qwen OpenAI caption failed: {e}")
