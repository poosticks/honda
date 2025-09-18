import json
from honda.prompter_openai import PromptVariantGenerator

cfg = {
    "backend": "openai_compat",
    "max_variants": 5,
    "openai": {
        "api_base": "http://127.0.0.1:1234/v1",
        "api_key": "",
        "model": "qwen2.5-vl-3b-instruct",
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 128,
        "request_timeout_s": 30,
    },
}
gen = PromptVariantGenerator(cfg)
arr = gen.variants_from_caption("a woman with long hair and a necklace, in a photograph")
print(json.dumps({"count": len(arr), "variants": arr}, ensure_ascii=False, indent=2))
