# honda/cli.py
from __future__ import annotations
import json
import pathlib
import random
import time
from typing import Any, Dict, List, Tuple

import typer
from PIL import Image
import yaml

from .comfy_client import ComfyClient
from .metrics import CompositeScorer
from .prompter_openai import PromptVariantGenerator
from .diff_qwen_openai import QwenVisualDiff
from .caption_qwen_openai import ImageCaptionerQwenOpenAI

app = typer.Typer(add_completion=False)


def _load_cfg(config: str | None) -> Dict[str, Any]:
    p = pathlib.Path(config or "config.yaml")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_txt2img_workflow(cfg: Dict[str, Any], prompt: str, negative: str, seed: int) -> Dict[str, Any]:
    comfy = cfg["comfy"]
    return {
        "0": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": comfy["ckpt_name"]}},
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["0", 1]}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["0", 1]}},
        "3": {"class_type": "EmptyLatentImage", "inputs": {
            "width": int(comfy["width"]), "height": int(comfy["height"]), "batch_size": int(comfy["batch_size"])
        }},
        "4": {"class_type": "KSampler", "inputs": {
            "seed": int(seed),
            "steps": int(comfy["sd_steps"]),
            "cfg": float(comfy["cfg_scale"]),
            "sampler_name": comfy["sampler"],
            "scheduler": comfy["scheduler"],
            "denoise": 1.0,
            "model": ["0", 0],
            "positive": ["1", 0],
            "negative": ["2", 0],
            "latent_image": ["3", 0]
        }},
        "5": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0], "vae": ["0", 2]}},
        "6": {"class_type": "SaveImage", "inputs": {"filename_prefix": comfy["filename_prefix"], "images": ["5", 0]}},
    }


def _select_best(scorer: CompositeScorer, ref_path: pathlib.Path, cand_paths: List[pathlib.Path]) -> Tuple[pathlib.Path, Dict[str, float]]:
    ref_img = Image.open(ref_path).convert("RGB")
    best_path = None
    best_metrics = None
    best_score = -1e9
    for p in cand_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        m = scorer.score_pair(ref_img, img)
        score = m.get("combined", 0.0)
        if score > best_score:
            best_score = score
            best_path = p
            best_metrics = m
    return best_path, best_metrics or {"lpips": 1.0, "clip_cos": 0.0, "combined": 0.0}


@app.command()
def generate(
    prompt: str = typer.Option(..., help="Positive prompt."),
    seeds: int = typer.Option(3, help="How many seeds to try."),
    config: str = typer.Option("config.yaml", help="Path to config YAML."),
    verbose: bool = typer.Option(False, help="Extra JSON logging."),
):
    cfg = _load_cfg(config)
    comfy = cfg["comfy"]
    paths = cfg["paths"]

    client = ComfyClient(comfy["host"], int(comfy["port"]))
    repo_outputs_dir = pathlib.Path(paths["outputs_dir"]).resolve()
    repo_outputs_dir.mkdir(parents=True, exist_ok=True)

    negative = cfg["prompter"]["default_negative"] if cfg["prompter"].get("inject_negative", True) else ""

    results = []
    for s in range(1, seeds + 1):
        wf = _build_txt2img_workflow(cfg, prompt, negative, seed=s)
        resp = client.queue_prompt(wf)
        hist = client.await_prompt(resp["prompt_id"], poll_interval_s=cfg["io"]["poll_interval_s"],
                                   timeout_s=cfg["io"]["history_timeout_s"])

        saved = client.download_all_outputs(hist, repo_outputs_dir)
        results.append({"seed": s, "prompt_id": resp.get("prompt_id"), "saved": [str(p) for p in saved]})

    if verbose:
        typer.echo(json.dumps({"debug": "resolved_workflow"}))
    typer.echo(json.dumps({"ok": True, "results": results}))


@app.command()
def converge(
    ref: pathlib.Path = typer.Option(..., exists=True, readable=True, help="Reference image path."),
    steps: int = typer.Option(30, help="Comfy sampler steps (cfg.comfy.sd_steps will be overridden)."),
    budget: int = typer.Option(30, help="Total images to render."),
    device: str = typer.Option("cpu", help="Scoring device (cpu/cuda:0)."),
    config: str = typer.Option("config.yaml", help="Path to config YAML."),
):
    cfg = _load_cfg(config)
    # reflect CLI overrides
    cfg["convergence"]["total_budget"] = int(budget)
    cfg["convergence"]["device"] = device
    cfg["comfy"]["sd_steps"] = int(steps)

    # 1) caption with Qwen-VL (for init base prompt)
    cap = ImageCaptionerQwenOpenAI(cfg)
    caption_text = cap.caption(str(ref))
    typer.echo(json.dumps({"stage": "init", "caption": caption_text, "source": "qwen_vl"}))

    # 2) prompter and scorer and visual diff
    prompter = PromptVariantGenerator(cfg["prompter"])
    scorer = CompositeScorer(cfg["scoring"])
    qdiff = QwenVisualDiff(cfg)

    # 3) I/O
    comfy = cfg["comfy"]
    paths = cfg["paths"]
    client = ComfyClient(comfy["host"], int(comfy["port"]))
    repo_outputs_dir = pathlib.Path(paths["outputs_dir"]).resolve()
    repo_outputs_dir.mkdir(parents=True, exist_ok=True)

    # optional: where Comfy writes (for fallback)
    comfy_out_dir = pathlib.Path(paths.get("comfy_output_dir", "") or "").resolve() if paths.get("comfy_output_dir") else None

    # loop params
    conv = cfg["convergence"]
    iterations = int(conv["iterations"])
    seeds_per_round = int(conv["seeds_per_round"])
    total_budget = int(conv["total_budget"])
    remaining = total_budget

    # negatives control
    default_negative = cfg["prompter"]["default_negative"]
    inject_negative = bool(cfg["prompter"].get("inject_negative", True))
    negative = default_negative if inject_negative else ""

    # carry forward settings
    carry_n = int(cfg["prompter"].get("carry_forward_modifiers", 1))

    best_overall = None  # dict with keys: path, prompt, metrics

    round_i = 0
    while remaining > 0 and round_i < iterations:
        # If we already have a winner from a previous round, diff it vs reference and build feedback
        if best_overall:
            fb = qdiff.diff(str(ref), str(best_overall["path"]))
        else:
            fb = {"add": [], "remove": [], "style": [], "neg": [], "summary": ""}

        # Update negatives for this round
        if inject_negative:
            extra_negs = sorted(set((fb.get("neg") or []) + (fb.get("remove") or [])))
            negative = default_negative + (", " + ", ".join(extra_negs) if extra_negs else "")

        # Ask LLM for feedback-aware variants
        base_prompt = (best_overall or {}).get("prompt") or caption_text
        variants = prompter.generate_variants(
            base_prompt,
            max_variants=int(cfg["prompter"]["max_variants"]),
            feedback=fb if best_overall else None,
        )
        typer.echo(json.dumps({"stage": "prompt", "count": len(variants)}))

        # Simple winner-take-most seed allocation:
        V = max(1, len(variants))
        per_round = min(seeds_per_round * V, remaining)
        # allocate seeds: first variant gets 50%, second 30%, rest share 20%
        plan: List[Tuple[str, int]] = []
        if V == 1:
            plan.append((variants[0], per_round))
        else:
            fifty = max(1, int(0.5 * per_round))
            thirty = max(0, int(0.3 * per_round))
            rest = max(0, per_round - fifty - thirty)
            plan.append((variants[0], fifty))
            if V >= 2 and thirty > 0:
                plan.append((variants[1], thirty))
            if rest > 0 and V > 2:
                # spread across the remaining variants
                q = rest // (V - 2)
                r = rest % (V - 2)
                for i, v in enumerate(variants[2:], start=2):
                    plan.append((v, q + (1 if i - 2 < r else 0)))

        # Render and score
        round_candidates: List[Tuple[pathlib.Path, Dict[str, float], str]] = []  # (image_path, metrics, prompt)
        for v_prompt, n_seeds in plan:
            for s in range(n_seeds):
                wf = _build_txt2img_workflow(cfg, v_prompt, negative, seed=random.randint(1, 2**31 - 1))
                resp = client.queue_prompt(wf)
                hist = client.await_prompt(resp["prompt_id"], poll_interval_s=cfg["io"]["poll_interval_s"],
                                           timeout_s=cfg["io"]["history_timeout_s"])
                saved = client.download_all_outputs(hist, repo_outputs_dir, disk_fallback=comfy_out_dir)
                for p in saved:
                    try:
                        mp = pathlib.Path(p).resolve()
                        # score on the fly to emit telemetry
                        ref_img = Image.open(ref).convert("RGB")
                        cand_img = Image.open(mp).convert("RGB")
                        m = scorer.score_pair(ref_img, cand_img)
                        round_candidates.append((mp, m, v_prompt))
                        typer.echo(json.dumps({"stage": "scored", "candidate": str(mp), **m}))
                    except Exception:
                        continue
                remaining -= 1
                if remaining <= 0:
                    break
            if remaining <= 0:
                break

        # Select this round’s best by combined
        if round_candidates:
            rc_sorted = sorted(round_candidates, key=lambda t: t[1].get("combined", 0.0), reverse=True)
            best_path, best_metrics, best_prompt = rc_sorted[0]
            # log best-of-round
            typer.echo(json.dumps({"stage": "best", "path": str(best_path), "metrics": best_metrics}))

            # Carry forward
            best_overall = {"path": best_path, "prompt": best_prompt, "metrics": best_metrics}

        round_i += 1

    # Final result
    out = {"stage": "done"}
    if best_overall:
        out["best"] = str(best_overall["path"])
        out["metrics"] = {"lpips": best_overall["metrics"].get("lpips")}
    typer.echo(json.dumps(out))
