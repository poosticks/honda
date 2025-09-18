import json, yaml, pathlib, time
from honda.comfy_client import ComfyClient

CFG = pathlib.Path("config.yaml")

def build_workflow(cfg, positive, negative, seed):
    comfy = cfg["comfy"]
    return {
        "0": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": comfy["ckpt_name"]}},
        "1": {"class_type": "CLIPTextEncode",
              "inputs": {"text": positive, "clip": ["0", 1]}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["0", 1]}},
        "3": {"class_type": "EmptyLatentImage",
              "inputs": {"width": comfy["width"], "height": comfy["height"], "batch_size": comfy["batch_size"]}},
        "4": {"class_type": "KSampler",
              "inputs": {"seed": int(seed), "steps": comfy["sd_steps"], "cfg": float(comfy["cfg_scale"]),
                         "sampler_name": comfy["sampler"], "scheduler": comfy["scheduler"], "denoise": 1.0,
                         "model": ["0", 0], "positive": ["1", 0], "negative": ["2", 0], "latent_image": ["3", 0]}},
        "5": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0], "vae": ["0", 2]}},
        "6": {"class_type": "SaveImage",
              "inputs": {"images": ["5", 0],
                         "filename_prefix": comfy["filename_prefix"],
                         "subfolder": comfy.get("output_subfolder", "")}},
    }

def _resolve_output_dir(cfg):
    explicit = cfg.get("comfy", {}).get("output_dir")
    if explicit:
        return pathlib.Path(explicit).resolve()

    here = pathlib.Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if parent.name.lower() == "comfyui":
            guess = parent / "output"
            if guess.exists():
                return guess.resolve()

    for p in [here.parent / "output", here.parent.parent / "output", here.parent.parent.parent / "output"]:
        if p.exists():
            return p.resolve()
    return (here.parent / "output").resolve()

def main():
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))

    positive = "smoke test portrait, simple"
    negative = cfg.get("prompter", {}).get("default_negative", "")
    comfy_output_dir = _resolve_output_dir(cfg)
    subfolder = cfg["comfy"].get("output_subfolder", "")
    prefix = cfg["comfy"]["filename_prefix"]

    client = ComfyClient(
        cfg["comfy"]["host"],
        int(cfg["comfy"]["port"]),
        request_timeout_s=120.0,
        comfy_output_dir=comfy_output_dir,
    )

    wf = build_workflow(cfg, positive, negative, seed=1)

    t0 = time.time()
    resp = client.queue_prompt(wf)
    pid = resp.get("prompt_id")

    entry = client.await_prompt(
        pid,
        poll_interval_s=float(cfg["io"]["poll_interval_s"]),
        timeout_s=float(cfg["io"]["history_timeout_s"]),
    )

    outputs = (entry or {}).get("outputs", {}) if isinstance(entry, dict) else {}
    images_reported = []
    for node_id, node in (outputs or {}).items():
        for img in node.get("images", []):
            images_reported.append({
                "node_id": node_id,
                "filename": img.get("filename"),
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output")
            })

    repo_outputs_dir = pathlib.Path(cfg["paths"]["outputs_dir"]).resolve()
    saved, error = [], None
    try:
        # 1) Try canonical (/view -> disk) based on history-reported filenames
        saved = client.download_all_outputs(entry, repo_outputs_dir)
        if not saved:
            # 2) As a safety net, collect by prefix since t0 (e.g., if managers renamed files)
            saved = client.collect_recent_by_prefix(
                output_prefix=prefix,
                dst_dir=repo_outputs_dir,
                subfolder=subfolder,
                window_s=max(60.0, time.time() - t0 + 10.0),
                max_files=8,
            )
            if not saved:
                raise RuntimeError("No files were saved via history filenames or recent-by-prefix collection.")
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    print(json.dumps({
        "ok": error is None,
        "prompt_id": pid,
        "comfy_output_dir": str(comfy_output_dir),
        "forced_subfolder": subfolder,
        "history_entry_has_outputs": bool(outputs),
        "images_reported_by_history": images_reported,
        "saved_to_repo_outputs": [str(p) for p in saved],
        "error": error
    }))

if __name__ == "__main__":
    main()
