# honda/converge.py
from __future__ import annotations
import json, time, uuid, os, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple

from .prompt_llm import LLMPrompter
from .prompt_blocks import PromptManager
from .metrics import CompositeScorer
from . import caption_blip
from .comfy_client import ComfyClient

# NEW: OpenAI-compatible prompter (LM Studio)
try:
    from .prompt_openai import OpenAICompatPrompter
    _HAS_OPENAI_PROMPTER = True
except Exception:
    _HAS_OPENAI_PROMPTER = False


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

@dataclass
class GenResult:
    prompt: str
    seed: int
    comfy_path: Path
    local_path: Path

def _copy_output(src: Path, dst_dir: Path, overwrite: bool) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists() and not overwrite:
        return dst
    shutil.copy2(src, dst)
    return dst

def _wait_for_image(comfy_out_dir: Path, prefix: str, timeout_s: float, poll_s: float) -> Optional[Path]:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        hits = sorted(comfy_out_dir.glob(f"{prefix}*.png"))
        if hits:
            return hits[-1]
        time.sleep(poll_s)
    return None


def run_convergence(ref_path: str, cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Convergence loop:
      1) Ask LLM prompter for {base, variants[], negatives[]}
      2) Build candidates = [base] + [base + ", variant"]
      3) Generate seeds per candidate via Comfy
      4) Score vs reference (LPIPS+CLIP)
      5) Pick best; carry tail modifiers to next step
      6) Log JSONL in runs/
    """
    paths = cfg["paths"]
    comfy_cfg = cfg["comfy"]
    conv_cfg = cfg["convergence"]
    prompt_cfg = cfg["prompter"]
    io_cfg = cfg["io"]
    scoring_cfg = cfg["scoring"]

    outputs_dir = Path(paths["outputs_dir"]).resolve()
    runs_dir = Path(paths["runs_dir"]).resolve()
    comfy_out_dir = Path(os.environ.get("COMFY_OUTPUT_DIR", r"C:\ComfyUI\ComfyUI\output")).resolve()
    _ensure_dir(outputs_dir); _ensure_dir(runs_dir)

    client = ComfyClient(comfy_cfg["host"], comfy_cfg["port"], timeout_s=io_cfg["history_timeout_s"])

    # Caption first (BLIP default for stability)
    captioner = caption_blip.ImageCaptioner(cfg)
    caption = captioner.caption_image(ref_path)
    yield {"stage": "init", "caption": caption}

    # Choose prompter
    backend = prompt_cfg.get("backend", "openai_compat")
    if backend == "openai_compat" and _HAS_OPENAI_PROMPTER:
        prompter = OpenAICompatPrompter(cfg)
    elif backend == "qwen_vl":
        prompter = LLMPrompter(cfg)  # transformers-based local Qwen (if installed)
    else:
        # fallback is implemented inside the LLMPrompter
        prompter = LLMPrompter(cfg)

    if getattr(prompter, "last_info", None):
        yield {"stage": "info", **prompter.last_info}

    # Prompt manager with carry-forward
    manager = PromptManager(caption, carry_forward_modifiers=prompt_cfg.get("carry_forward_modifiers", 1))

    # Scorer (resizes images consistently to fix shape mismatches)
    scorer = CompositeScorer(
        device=scoring_cfg.get("device", "cpu"),
        use_lpips=scoring_cfg.get("use_lpips", True),
        use_clip=scoring_cfg.get("use_clip", True),
        clip_model=scoring_cfg.get("clip_model", "ViT-B/32"),
        resize=scoring_cfg.get("resize", 256),
    )

    run_id = uuid.uuid4().hex[:8]
    run_path = runs_dir / f"run_{run_id}.jsonl"
    best_path = None
    best_metrics = {}

    with open(run_path, "a", encoding="utf-8") as logf:
        iterations = int(conv_cfg.get("iterations", 3))
        total_budget = int(conv_cfg.get("total_budget", 24))
        seeds_per_round = int(conv_cfg.get("seeds_per_round", 5))

        injected_neg = []
        if prompt_cfg.get("inject_negative", True):
            dn = prompt_cfg.get("default_negative", "")
            if dn:
                injected_neg = [dn]

        budget_used = 0
        best_prompt = manager.base

        for step in range(iterations):
            # Ask LLM for structured proposals
            props = prompter.propose(ref_path, best_prompt, prompt_cfg.get("max_variants", 7))
            base = props.get("base", best_prompt).strip() or best_prompt
            negs = injected_neg or props.get("negatives", [])
            variants = props.get("variants", [])

            yield {"stage": "prompt", "count": len(variants) + 1, "base": base, "variants": variants, "negs": negs}
            logf.write(json.dumps({"event": "variants", "step": step, "base": base, "variants": variants, "negatives": negs}) + "\n"); logf.flush()

            # Build candidate prompts
            candidates = manager.expand_with_variants(base, variants)

            # Generate & score
            step_scores: List[Tuple[str, float, float, float, str]] = []
            for cand in candidates:
                for s in range(seeds_per_round):
                    if budget_used >= total_budget:
                        break
                    seed_val = s + 1
                    prefix = f'{comfy_cfg.get("filename_prefix","honda_")}{uuid.uuid4().hex[:8]}_'
                    wf = {
                        "0": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": comfy_cfg["ckpt_name"]}},
                        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": cand, "clip": ["0", 1]}},
                        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ", ".join(negs), "clip": ["0", 1]}},
                        "3": {"class_type": "EmptyLatentImage", "inputs": {"width": comfy_cfg["width"], "height": comfy_cfg["height"], "batch_size": 1}},
                        "4": {
                            "class_type": "KSampler",
                            "inputs": {
                                "model": ["0", 0],
                                "seed": seed_val,
                                "steps": comfy_cfg["sd_steps"],
                                "cfg": comfy_cfg["cfg_scale"],
                                "sampler_name": comfy_cfg["sampler"],
                                "scheduler": comfy_cfg["scheduler"],
                                "positive": ["1", 0],
                                "negative": ["2", 0],
                                "latent_image": ["3", 0],
                                "denoise": 1.0,
                            },
                        },
                        "5": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0], "vae": ["0", 2]}},
                        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": prefix}},
                    }

                    try:
                        _ = client.submit(wf)
                        img = _wait_for_image(comfy_out_dir, prefix, io_cfg["history_timeout_s"], io_cfg["poll_interval_s"])
                        if not img:
                            yield {"stage": "warning", "message": "No image detected after generation", "prefix": prefix}
                            logf.write(json.dumps({"event": "no_image", "step": step, "prompt": cand, "seed": seed_val, "prefix": prefix}) + "\n"); logf.flush()
                            continue

                        local = _copy_output(Path(img), Path(paths["outputs_dir"]), io_cfg.get("overwrite_existing", False))
                        budget_used += 1
                        yield {"stage": "generated", "prompt": cand, "seed": seed_val, "path": str(img)}
                        logf.write(json.dumps({"event": "generated", "step": step, "prompt": cand, "seed": seed_val, "comfy_path": str(img), "local_path": str(local)}) + "\n"); logf.flush()

                        metrics = scorer.score_pair(ref_path, str(local))
                        step_scores.append((cand, metrics.get("lpips", 9.9), metrics.get("clip", 0.0), metrics.get("combined", 0.0), str(local)))
                        yield {"stage": "scored", "candidate": str(local), **{k: v for k, v in metrics.items()}}
                        logf.write(json.dumps({"event": "scored", "step": step, "candidate": str(local), **metrics}) + "\n"); logf.flush()

                    except Exception as e:
                        yield {"stage": "error", "when": "generate", "details": str(e)}
                        logf.write(json.dumps({"event": "error", "step": step, "phase": "generate", "error": str(e)}) + "\n"); logf.flush()

                if budget_used >= total_budget:
                    break

            if not step_scores:
                continue

            step_scores.sort(key=lambda x: x[3], reverse=True)  # higher combined is better
            best_prompt = step_scores[0][0]
            best_path = step_scores[0][4]
            best_metrics = {"lpips": step_scores[0][1], "clip": step_scores[0][2], "combined": step_scores[0][3]}
            yield {"stage": "best", "path": best_path, "metrics": best_metrics}
            logf.write(json.dumps({"event": "best", "step": step, "prompt": best_prompt, "path": best_path, "metrics": best_metrics}) + "\n"); logf.flush()

            # Carry forward tail modifiers
            best_prompt = manager.refine_with_best(best_prompt)

        yield {"stage": "done", "best": best_path, "metrics": best_metrics}
        logf.write(json.dumps({"event": "done", "best": best_path, "metrics": best_metrics}) + "\n"); logf.flush()
