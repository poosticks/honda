import os
import json
from copy import deepcopy
from .comfy_client import ComfyClient
from .caption_blip import ImageCaptioner
from .prompt_blocks import PromptManager
from .doe import DOEModule
from . import metrics
from . import utils

def run_convergence(reference_image: str, config: dict):
    """
    Full convergence loop. Yields JSON-able dicts (per image, per-iter, final).
    """
    # 1) Caption
    device = config.get("device", "cuda")
    blip_model = config.get("blip_model", "Salesforce/blip-image-captioning-large")
    captioner = ImageCaptioner(device=device, blip_model=blip_model)
    caption = captioner.caption(reference_image)

    prompt_manager = PromptManager(initial_caption=caption)
    negative_prompt = prompt_manager.get_negative_prompt()

    # 2) DOE seeds
    seed_count = int(config.get("seeds", 30))
    base_seeds = list(range(1, seed_count + 1))
    doe = DOEModule(seeds=base_seeds)

    # 3) Comfy
    host = config.get("comfy_host", "127.0.0.1")
    port = int(config.get("comfy_port", 8188))
    client = ComfyClient(host=host, port=port)

    wf_path = config.get("workflow")
    if not wf_path or not os.path.exists(wf_path):
        raise RuntimeError(f"Workflow JSON not found: {wf_path}")
    base_workflow = json.loads(open(wf_path, "r", encoding="utf-8").read())

    # Override checkpoint (RESPECT your YAML)
    ckpt_name = config.get("ckpt_name")
    if ckpt_name:
        base_workflow["0"]["inputs"]["ckpt_name"] = ckpt_name

    # Discover & normalize sampler/scheduler to avoid 400s
    selected_sampler = utils.normalize_sampler(
        config.get("sampler", "euler_ancestral"), client.get_samplers()
    )
    selected_scheduler = utils.normalize_scheduler(
        config.get("scheduler", "normal"), client.get_schedulers()
    )

    # SD node params (RESPECT steps/cfg from your YAML)
    base_workflow["4"]["inputs"]["sampler_name"] = selected_sampler
    base_workflow["4"]["inputs"]["scheduler"] = selected_scheduler
    base_workflow["4"]["inputs"]["steps"] = int(config.get("sd_steps", 20))
    base_workflow["4"]["inputs"]["cfg"] = float(config.get("cfg_scale", 8.0))
    base_workflow["3"]["inputs"]["batch_size"] = 1  # 6GB VRAM safety

    # Convergence loop limits
    max_iters = int(config.get("iters", 3))
    max_images_total = config.get("budget", None)
    max_images_total = int(max_images_total) if max_images_total is not None else None
    use_dream = bool(config.get("use_dreamsim", False))
    output_dir = config.get("output_dir", "outputs")
    utils.ensure_dir(output_dir)

    images_generated = 0
    best_prompt, best_score = None, None

    for iteration in range(1, max_iters + 1):
        prompt_list = prompt_manager.generate_prompt_variants()
        remaining = None if max_images_total is None else max_images_total - images_generated
        batch_pairs = doe.plan_experiments(prompt_list, max_images=remaining)
        if not batch_pairs:
            break

        prompt_scores = {}

        for prompt_text, seed in batch_pairs:
            wf = deepcopy(base_workflow)
            wf["1"]["inputs"]["text"] = prompt_text
            wf["2"]["inputs"]["text"] = negative_prompt
            wf["4"]["inputs"]["seed"] = int(seed)

            prompt_id = client.submit(wf)
            history = client.wait_for_completion(prompt_id)
            outputs = history.get("outputs", {}) or {}
            for node_id, outdata in outputs.items():
                if outdata.get("images"):
                    for img_info in outdata["images"]:
                        fname = img_info["filename"]; subf = img_info["subfolder"]; ftype = img_info["type"]
                        img_bytes = client.get_image(fname, subfolder=subf, folder_type=ftype)
                        out_path = os.path.join(output_dir, fname)
                        with open(out_path, "wb") as f:
                            f.write(img_bytes)

                        scores = metrics.compute_composite_score(out_path, reference_image, use_dream=use_dream, device=device)
                        score_val = scores["score"]
                        yield {
                            "iter": iteration,
                            "prompt": prompt_text,
                            "negative": negative_prompt,
                            "seed": int(seed),
                            "output": out_path,
                            "lpips": scores["lpips"],
                            "clip_sim": scores["clip_sim"],
                            "dream_sim": scores.get("dream_sim"),
                            "score": score_val
                        }

                        if prompt_text not in prompt_scores or score_val > prompt_scores[prompt_text]:
                            prompt_scores[prompt_text] = score_val
                        images_generated += 1
                        if max_images_total is not None and images_generated >= max_images_total:
                            break
            if max_images_total is not None and images_generated >= max_images_total:
                break

        if not prompt_scores:
            break

        iter_best_prompt = max(prompt_scores, key=lambda p: prompt_scores[p])
        iter_best_score = prompt_scores[iter_best_prompt]
        yield {"iter": iteration, "summary": {"best_prompt": iter_best_prompt, "best_score": iter_best_score}}

        if best_score is None or iter_best_score > best_score:
            best_score, best_prompt = iter_best_score, iter_best_prompt

        if (max_images_total is not None and images_generated >= max_images_total) or (iteration >= max_iters):
            break

        prompt_manager.update_from_scores(prompt_scores)
        negative_prompt = prompt_manager.get_negative_prompt()

    yield {"converged": True, "final_prompt": best_prompt, "score": best_score}
