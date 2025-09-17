from pathlib import Path
from typing import Optional, List
from copy import deepcopy
import time
import json
import typer

from . import utils
from .comfy_client import ComfyClient
from .caption_blip import ImageCaptioner
from . import converge as converge_mod
from . import metrics

app = typer.Typer(help="Honda: Convergent image generation pipeline")

def _norm(x: Optional[str]) -> str:
    return (x or "").strip().lower()

def _resolve_ref(ref_arg: Optional[Path], ref_opt: Optional[Path]) -> Path:
    path = ref_opt or ref_arg
    if path is None:
        raise typer.BadParameter("Provide the reference image as a positional argument or with --ref.")
    if not path.exists():
        raise typer.BadParameter(f"Reference image not found: {path}")
    return path

def _map_ckpt_name(cfg: dict, client: ComfyClient, requested: Optional[str]) -> str:
    """Try server discovery; if empty, fall back to disk scan under ComfyUI/models/checkpoints."""
    req = (requested or "").strip()
    server = client.get_checkpoints()
    disk = utils.discover_checkpoints_on_disk(cfg)

    # Combine (server first for fidelity)
    candidates = server + [x for x in disk if x not in server]

    if not candidates:
        # no discovery available: return as-is
        return req

    # exact
    if req in candidates:
        return req

    # case-insensitive exact
    for s in candidates:
        if _norm(s) == _norm(req):
            if s != req:
                typer.echo(json.dumps({"info": "ckpt_mapped", "requested": req, "selected": s}))
            return s

    # endswith (handles subfolders)
    ends = [s for s in candidates if _norm(s).endswith(_norm(req))]
    if len(ends) == 1:
        typer.echo(json.dumps({"info": "ckpt_mapped", "requested": req, "selected": ends[0]}))
        return ends[0]
    if len(ends) > 1:
        typer.echo(json.dumps({"error": "ckpt_ambiguous", "requested": req, "candidates": ends}))
        raise typer.Exit(2)

    # no match → show list for user to pick/update in YAML
    typer.echo(json.dumps({"error": "ckpt_not_found", "requested": req, "available": candidates}))
    raise typer.Exit(2)

@app.command()
def caption(
    ref_arg: Optional[Path] = typer.Argument(None, help="Reference image path"),
    ref: Optional[Path] = typer.Option(None, "--ref", "-r", help="Reference image path"),
    config_path: Path = typer.Option("config.yaml", "--config", help="Path to config file"),
):
    cfg = utils.load_config(str(config_path))
    image_path = _resolve_ref(ref_arg, ref)
    device = cfg.get("device", "cuda")
    model_name = cfg.get("blip_model", "Salesforce/blip-image-captioning-large")
    captioner = ImageCaptioner(device=device, blip_model=model_name)
    result = {"caption": captioner.caption(str(image_path))}
    typer.echo(json.dumps(result))

@app.command()
def generate(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Positive prompt text"),
    negative: str = typer.Option("", "--negative", "-n", help="Negative prompt text"),
    seeds: int = typer.Option(1, "--seeds", "-s", help="Number of seeds (images) to generate"),
    output: Path = typer.Option(Path("outputs"), "--output", "-o", help="Output directory to save images"),
    config_path: Path = typer.Option("config.yaml", "--config", help="Path to config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print resolved workflow and diagnostics"),
):
    cfg = utils.load_config(str(config_path))
    cfg["output_dir"] = str(output)

    client = ComfyClient(host=cfg.get("comfy_host", "127.0.0.1"), port=cfg.get("comfy_port", 8188))

    # Checkpoint mapping (server → disk fallback)
    requested_ckpt = cfg.get("ckpt_name") or ""
    resolved_ckpt = _map_ckpt_name(cfg, client, requested_ckpt)

    # Sampler/scheduler mapping
    server_samplers = client.get_samplers()
    server_schedulers = client.get_schedulers()
    requested_sampler = cfg.get("sampler", "euler_ancestral")
    requested_scheduler = cfg.get("scheduler", "normal")
    selected_sampler = utils.normalize_sampler(requested_sampler, server_samplers)
    selected_scheduler = utils.normalize_scheduler(requested_scheduler, server_schedulers)
    if _norm(requested_sampler) != _norm(selected_sampler):
        typer.echo(json.dumps({"info": "sampler_mapped", "requested": requested_sampler, "selected": selected_sampler}))
    if _norm(requested_scheduler) != _norm(selected_scheduler):
        typer.echo(json.dumps({"info": "scheduler_mapped", "requested": requested_scheduler, "selected": selected_scheduler}))

    # Load workflow template
    wf_path = cfg.get("workflow")
    if not wf_path or not Path(wf_path).exists():
        typer.echo(json.dumps({"error": "workflow_not_found", "wf_path": wf_path}))
        raise typer.Exit(1)

    try:
        base_workflow = json.loads(Path(wf_path).read_text(encoding="utf-8"))
    except Exception as e:
        typer.echo(json.dumps({"error": "invalid_workflow_json", "wf_path": wf_path, "details": str(e)}))
        raise typer.Exit(2)

    # Quick shape check
    for key in ("0", "4", "6"):
        if key not in base_workflow:
            typer.echo(json.dumps({"error": "invalid_workflow_shape", "missing": key}))
            raise typer.Exit(2)

    # Apply config (respecting your values)
    base_workflow["0"]["inputs"]["ckpt_name"] = resolved_ckpt
    base_workflow["4"]["inputs"]["sampler_name"] = selected_sampler
    base_workflow["4"]["inputs"]["scheduler"] = selected_scheduler
    base_workflow["4"]["inputs"]["steps"] = int(cfg.get("sd_steps", 30))
    base_workflow["4"]["inputs"]["cfg"] = float(cfg.get("cfg_scale", 5.0))
    base_workflow["3"]["inputs"]["batch_size"] = 1  # keep VRAM-friendly on 6GB

    if verbose:
        digest = {
            "nodes": len(base_workflow),
            "ksampler": base_workflow["4"]["class_type"],
            "saveimage": base_workflow["6"]["class_type"],
            "ckpt_name": base_workflow["0"]["inputs"].get("ckpt_name"),
            "sampler_name": base_workflow["4"]["inputs"].get("sampler_name"),
            "scheduler": base_workflow["4"]["inputs"].get("scheduler"),
            "steps": base_workflow["4"]["inputs"].get("steps"),
            "cfg": base_workflow["4"]["inputs"].get("cfg"),
        }
        typer.echo(json.dumps({"debug": "resolved_workflow", "digest": digest}))

    # Output pickup config
    comfy_out_dir = utils.find_comfy_output_dir(cfg)
    filename_prefix = base_workflow.get("6", {}).get("inputs", {}).get("filename_prefix", "comfy_output")
    history_timeout = int(cfg.get("history_timeout_s", 90))

    # Run seeds
    for i in range(seeds):
        seed_val = i + 1
        wf = deepcopy(base_workflow)
        wf["1"]["inputs"]["text"] = prompt
        wf["2"]["inputs"]["text"] = negative
        wf["4"]["inputs"]["seed"] = seed_val

        start_ts = time.time()
        pid = client.submit(wf)

        try:
            history = client.wait_for_completion(pid, max_wait_s=history_timeout)
        except TimeoutError:
            history = None

        images_emitted = 0
        if isinstance(history, dict):
            outputs = history.get("outputs", {}) or {}
            for _, outdata in outputs.items():
                for img_info in (outdata.get("images") or []):
                    fname = img_info["filename"]; subf = img_info["subfolder"]; ftype = img_info["type"]
                    img_bytes = client.get_image(fname, subfolder=subf, folder_type=ftype)
                    utils.ensure_dir(str(output))
                    out_path = Path(output) / fname
                    out_path.write_bytes(img_bytes)
                    typer.echo(json.dumps({"seed": seed_val, "prompt": prompt, "negative": negative, "output": str(out_path)}))
                    images_emitted += 1

        if images_emitted == 0 and comfy_out_dir and comfy_out_dir.exists():
            recent = utils.find_recent_outputs(comfy_out_dir, filename_prefix, since_ts=start_ts, limit=1)
            if recent:
                dst = utils.copy_into_outputs(recent[0], Path(output))
                typer.echo(json.dumps({
                    "info": "disk_fallback",
                    "reason": "history_missing_images",
                    "comfy_output": str(recent[0]),
                    "output": str(dst),
                    "seed": seed_val,
                    "prompt": prompt,
                    "negative": negative
                }))
                images_emitted = 1

        if images_emitted == 0:
            err = {
                "error": "no_images_detected",
                "details": {
                    "prompt_id": pid,
                    "history_timeout_s": history_timeout,
                    "comfy_output_dir": str(comfy_out_dir) if comfy_out_dir else None,
                    "filename_prefix": filename_prefix,
                    "hint": {
                        "ckpt_used": resolved_ckpt,
                        "sampler": base_workflow["4"]["inputs"]["sampler_name"],
                        "scheduler": base_workflow["4"]["inputs"]["scheduler"]
                    }
                }
            }
            typer.echo(json.dumps(err))
            raise typer.Exit(2)

@app.command()
def score(
    ref_arg: Optional[Path] = typer.Argument(None, help="Reference image path"),
    ref: Optional[Path] = typer.Option(None, "--ref", "-r", help="Reference image path"),
    candidates: List[Path] = typer.Argument(..., help="Generated image(s) to score (glob or paths)"),
    use_dreamsim: bool = typer.Option(False, "--use-dreamsim", help="Use DreamSim metric if available"),
    config_path: Path = typer.Option("config.yaml", "--config", help="Path to config file"),
):
    cfg = utils.load_config(str(config_path))
    ref_path = _resolve_ref(ref_arg, ref)
    files = utils.list_images([str(p) for p in candidates])
    if not files:
        typer.echo("No candidate images found for scoring.", err=True)
        raise typer.Exit(1)
    for img in files:
        scores = metrics.compute_composite_score(img, str(ref_path), use_dream=use_dreamsim, device=cfg.get("device", "cuda"))
        result = {"image": img, "lpips": scores["lpips"], "clip": scores["clip_sim"], "dreamsim": scores.get("dream_sim"), "score": scores["score"]}
        typer.echo(json.dumps(result))

@app.command()
def converge(
    ref_arg: Optional[Path] = typer.Argument(None, help="Reference image path"),
    ref: Optional[Path] = typer.Option(None, "--ref", "-r", help="Reference image path"),
    iters: int = typer.Option(3, "--steps", "-t", help="Max prompt refinement iterations (not SD steps)"),
    budget: Optional[int] = typer.Option(None, "--budget", "-b", help="Max total images to generate"),
    config_path: Path = typer.Option("config.yaml", "--config", help="Path to config file"),
):
    cfg = utils.load_config(str(config_path))
    ref_path = _resolve_ref(ref_arg, ref)
    cfg["iters"] = iters
    if budget is not None:
        cfg["budget"] = budget
    for result in converge_mod.run_convergence(str(ref_path), cfg):
        typer.echo(json.dumps(result))

if __name__ == "__main__":
    app(prog_name="honda")
