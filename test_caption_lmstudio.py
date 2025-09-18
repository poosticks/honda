# Place next to config.yaml (C:\ComfyUI\ComfyUI\user\default\workflows\honda)
import json, yaml, pathlib, sys
from honda.caption_qwen_openai import ImageCaptionerQwenOpenAI

CFG_PATH = pathlib.Path("config.yaml")
REF_PATH = pathlib.Path("reference/2.jpg")  # change if you want

def main():
    if not CFG_PATH.exists():
        print(json.dumps({"ok": False, "error": "config.yaml not found"}))
        sys.exit(1)
    if not REF_PATH.exists():
        print(json.dumps({"ok": False, "error": f"reference image not found: {REF_PATH}"}))
        sys.exit(1)

    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))

    # sanity: we want Qwen-VL via LM Studio
    backend = cfg.get("caption", {}).get("backend")
    if backend != "qwen_openai":
        print(json.dumps({"ok": False, "error": f"caption.backend is '{backend}', expected 'qwen_openai'"}))
        sys.exit(1)

    cap = ImageCaptionerQwenOpenAI(cfg)
    caption = cap.caption(str(REF_PATH))
    print(json.dumps({"ok": True, "backend": backend, "caption": caption}))

if __name__ == "__main__":
    main()
