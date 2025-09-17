# Honda Usage Guide

This document provides detailed command examples, expected outputs, troubleshooting tips, determinism notes, and revision history.

---

## CLI Overview

Honda provides the following subcommands:

- `honda caption` — Generate a text caption from a reference image.
- `honda generate` — Run txt2img generations through ComfyUI.
- `honda score` — Compare candidates against a reference image.
- `honda converge` — Run iterative convergence loops.

Run `honda --help` or `honda <command> --help` for available flags.

---

## Commands and Examples

### 1. Caption
Generate a BLIP caption for a reference image:

honda caption ref.png

**Output (JSON line):**

{"caption": "a simple photo of a duck"}

---

### 2. Generate

Run text-to-image through ComfyUI:

honda generate --prompt "a simple test photo" --seeds 1 --verbose

**Output (JSON line):**

{"info": "disk_fallback_early", "output": "outputs/comfy_output_00004_.png", "seed": 1}


* `--prompt` : Prompt text for image generation.
* `--seeds` : Number of seeds or explicit list of seed values.
* `--verbose` : Enable detailed debug logs.
* `--negative` : Negative prompt (optional).
* `--steps` : Override default step count (default: 30).
* `--cfg` : CFG guidance scale (default: 5.0).

---

### 3. Score

Compare generated images against a reference:

honda score --ref ref.png --candidates outputs/*.png

**Output (JSON lines):**

{"candidate": "outputs/comfy_output_00004_.png", "lpips": 0.32, "clip": 0.81, "composite": 0.56}

* Uses LPIPS + CLIP/DINO similarity.
* Outputs composite scores for ranking.

---

### 4. Converge

Run iterative convergence across several rounds:

honda converge --ref ref.png --steps 3 --budget 90

**Output (JSON lines):**

{"round": 1, "best_score": 0.61, "prompt": "a photo of a duck"}
{"round": 2, "best_score": 0.74, "prompt": "a naturalistic photo of a duck near water"}

* `--steps` : Number of convergence iterations.
* `--budget` : Total generations allowed across steps.

---

## Troubleshooting

### `no_images_detected`

ComfyUI generated the image but didn’t report it in `/history`.
Honda now uses *disk-first pickup* (`disk_fallback_early`) to detect images quickly.

Adjust in `config.yaml`:

history_timeout_s: 90
disk_grace_s: 10

---

### Sampler mismatch

You may see:

{"info": "sampler_mapped", "requested": "euler_a", "selected": "euler_ancestral"}

Honda maps deprecated samplers to valid ComfyUI samplers.

---

### Checkpoint not found

Ensure `ckpt_name` in `config.yaml` matches a model inside:

ComfyUI/models/checkpoints/

---

### Impact Pack warnings

Safe to ignore unless your workflow explicitly depends on Impact Pack custom nodes.

---

## Determinism Notes

* **Seeds:** Honda uses fixed seeds per run for reproducibility.
* **Torch non-determinism:** For bit-perfect results, set:

  set CUBLAS_WORKSPACE_CONFIG=:16:8
  set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

* **Tokenization:** SD1.5 truncates beyond 75 tokens.
* **Disk-first pickup:** Prevents CLI from stalling when ComfyUI doesn’t update `/history`.

---

## Revision History

* **v0.3.0 (2025-09-17)**

  * Added disk-first pickup (`disk_fallback_early`)
  * Reduced CLI pause time
  * Added `USAGE.md`

* **v0.2.0 (2025-09-15)**

  * Fixed CLI parsing issues
  * Added `--verbose` flag
  * Improved ComfyUI sampler mapping

* **v0.1.0 (2025-09-10)**

  * Initial release: caption, generate, score, converge

---