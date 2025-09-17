Got it. Here’s a full **README.md** draft that covers objectives, CLI usage, file coordination, and a revision history that highlights the issues we’ve worked through. I’ll keep the structure clean and practical, so it’s ready to drop into your repo.

---

# Honda CLI for ComfyUI Integration

## Overview

`honda` is a command-line interface (CLI) that coordinates prompts, workflows, and outputs between **ComfyUI** and a local image-captioning/generation stack.
It enables:

* Captioning reference images (via BLIP/CLIP).
* Generating images using ComfyUI workflows (txt2img / img2img).
* Managing reproducibility with configs, seeds, and workflow coordination.

The CLI provides an auditable workflow: you supply prompts and parameters, Honda prepares a JSON workflow for ComfyUI, submits it, and collects results into structured outputs.

---

## Objectives

1. **Workflow orchestration**

   * Coordinate prompt → workflow → ComfyUI server → image output.
   * Map sampler/scheduler names between CLI configs and ComfyUI’s available options.
   * Handle outputs consistently and fail gracefully if nodes misfire.

2. **Reference image captioning**

   * Extract captions from a reference image with BLIP.
   * Output JSON with the caption text for downstream processing.

3. **Reproducibility**

   * Use `config.yaml` to lock in checkpoint, sampler, scheduler, steps, CFG scale, seeds, and batch size.
   * Output JSON logs for each run, so results can be traced.

---

## CLI Usage

```bash
honda [COMMAND] [OPTIONS]
```

### Commands

#### `caption`

Caption a reference image using BLIP.

```bash
honda caption reference/2.jpg
```

**Arguments & Options:**

* `REF` (positional): Path to the reference image.
* `--config`: Path to YAML config file (default: `config.yaml`).

Output: JSON with the generated caption.

---

#### `generate`

Generate images via ComfyUI workflow.

```bash
honda generate --prompt "a simple test photo" --seeds 1
```

**Arguments & Options:**

* `--prompt TEXT` (required): Positive prompt text.
* `--negative TEXT`: Negative prompt (default: empty).
* `--seeds INT`: Number of seeds to iterate over (default: from config).
* `--config PATH`: Path to YAML config file (default: `config.yaml`).
* `--output DIR`: Output directory for images (default: `outputs`).
* `--verbose`: Print resolved workflow and debug information.

Output: Images saved in `outputs/` with prefix `comfy_output`.

---

## Configuration (`config.yaml`)

Defines defaults and ensures reproducibility.

Example:

```yaml
comfy_host: "127.0.0.1"
comfy_port: 8188
workflow: "workflows/txt2img_sd1.5.json"
ckpt_name: "cyberrealistic_v90.safetensors"
sampler: "euler_a"
scheduler: "normal"
sd_steps: 30
cfg_scale: 5.0
seeds: 1
batch_size: 1
width: 512
height: 512
```

---

## File Responsibilities

* `honda/cli.py`
  Main entrypoint, defines CLI commands via Typer.

* `honda/utils.py`
  YAML config loader, output directory helpers.

* `honda/comfy_client.py`
  HTTP client for ComfyUI REST API, handles submit/wait/result parsing.

* `honda/captioning.py`
  BLIP-based image captioning implementation.

* `config.yaml`
  Default configuration (checkpoint, sampler, scheduler, etc).

* `workflows/txt2img_sd1.5.json`
  Workflow template for txt2img (can be adapted).

* `outputs/`
  Directory where results are saved.

---

## Revision History

**v0.1.0 (Initial)**

* Added `caption` and `generate` commands.
* Base workflow JSON included.
* Config-driven sampler/scheduler/steps/cfg.

**v0.1.1**

* Fixed CLI mismatch: `--ref` replaced with positional argument.
* Corrected `app = typer.Typer()` declaration to fix `NameError`.

**v0.1.2**

* Handled sampler mismatch (`euler_a` → `euler_ancestral`).
* Added verbose mode with workflow digest.

**v0.1.3**

* Fixed YAML parsing errors from `*` alias markers.
* Added validation on workflow outputs (`no_images_detected`).

**Known Issues / Ongoing**

* Some workflows execute in ComfyUI but return **no image** via CLI.
* Requires validation of `SaveImage` node presence and proper node IDs in workflow.
* Potential mismatch between ComfyUI Impact Pack updates and CLI assumptions.
* Currently pinned to `torch>=2.0.0` for Python 3.13 compatibility; Torch 2.0.1 unavailable, requirement updated.

---

Would you like me to split the **README.md** and **OBJECTIVES.md** into two files (README for usage, OBJECTIVES for architecture/coordination explanation), or keep them consolidated?
