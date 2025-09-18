# test_prompter_lmstudio.py
import yaml, json
from honda.prompter_openai import PromptVariantGenerator

cfg = yaml.safe_load(open("config.yaml", "r"))
gen = PromptVariantGenerator(cfg["prompter"])
base = "A young woman with long blonde hair wearing a delicate necklace, standing outdoors on a sunny day."
vars = gen.generate_variants(base, max_variants=int(cfg["prompter"].get("max_variants", 7)))
print(json.dumps({"ok": True, "count": len(vars), "variants": vars}, ensure_ascii=False))
