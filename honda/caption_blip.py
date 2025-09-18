# honda/caption_blip.py
from __future__ import annotations

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


class ImageCaptioner:
    """
    Tiny BLIP wrapper: deterministic, CPU-ok, no surprises.
    """

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large",
                 device: str = "cpu", use_fast: bool = True):
        # Only use CUDA if requested and actually available
        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=use_fast)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def caption(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=48)
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return text.strip()
