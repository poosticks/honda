# honda/workflow.py
from __future__ import annotations
from typing import List, Optional


def _norm(name: Optional[str]) -> str:
    return (name or "").strip().lower().replace("-", "_")


def _maybe_map_sampler(requested: str, server_samplers: Optional[List[str]]) -> str:
    """
    Map legacy/alias sampler names to current Comfy names.
    IMPORTANT: We map known aliases even if server_samplers is empty.
    """
    req = _norm(requested)
    # Known aliases (apply even if discovery fails)
    known = {
        "euler_a": "euler_ancestral",
    }
    if req in known:
        return known[req]

    # If we have server samplers, try to match case/variant
    if server_samplers:
        lower = [s.lower().replace("-", "_") for s in server_samplers]
        if req in lower:
            return server_samplers[lower.index(req)]

    return requested


def _build_txt2img_workflow(
    ckpt_name: str,
    prompt: str,
    negative: str,
    sampler_name: str,
    scheduler: str,
    steps: int,
    cfg_scale: float,
    seed: int,
    width: int = 512,
    height: int = 512,
    batch_size: int = 1,
    filename_prefix: str = "honda",
) -> dict:
    """Return a ComfyUI workflow-compatible dict (SD1.5-like)."""
    wf = {
        "0": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt_name},
            "_meta": {"title": "Load SD1.5 Model"},
        },
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["0", 1]},
            "_meta": {"title": "CLIP Text Encode (Positive)"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative or "", "clip": ["0", 1]},
            "_meta": {"title": "CLIP Text Encode (Negative)"},
        },
        "3": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": batch_size},
            "_meta": {"title": "Empty Latent Image"},
        },
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["0", 0],
                "seed": int(seed),
                "steps": int(steps),
                "cfg": float(cfg_scale),
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "positive": ["1", 0],
                "negative": ["2", 0],
                "latent_image": ["3", 0],
                "denoise": 1.0,
            },
            "_meta": {"title": "KSampler"},
        },
        "5": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["4", 0], "vae": ["0", 2]},
            "_meta": {"title": "VAE Decode"},
        },
        "6": {
            "class_type": "SaveImage",
            "inputs": {"images": ["5", 0], "filename_prefix": filename_prefix},
            "_meta": {"title": "Save Image"},
        },
    }
    return wf


def _digest_for_workflow(wf: dict) -> dict:
    """Lightweight introspection for --verbose JSON."""
    return {
        "nodes": len(wf),
        "ksampler": wf.get("4", {}).get("class_type", ""),
        "saveimage": wf.get("6", {}).get("class_type", ""),
        "ckpt_name": wf.get("0", {}).get("inputs", {}).get("ckpt_name"),
        "sampler_name": wf.get("4", {}).get("inputs", {}).get("sampler_name"),
        "scheduler": wf.get("4", {}).get("inputs", {}).get("scheduler"),
        "steps": wf.get("4", {}).get("inputs", {}).get("steps"),
        "cfg": wf.get("4", {}).get("inputs", {}).get("cfg"),
    }
