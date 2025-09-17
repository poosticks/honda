import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaptioner:
    def __init__(self, device="cuda", blip_model="Salesforce/blip-image-captioning-large"):
        """
        Load BLIP for captioning. Falls back to BLIP-base if large fails (e.g., OOM).
        """
        want_cuda = (device == "cuda") and torch.cuda.is_available()
        self.device = torch.device("cuda" if want_cuda else "cpu")
        torch.set_grad_enabled(False)

        def _load(name):
            proc = BlipProcessor.from_pretrained(name)
            mdl = BlipForConditionalGeneration.from_pretrained(name).to(self.device)
            mdl.eval()
            return proc, mdl

        self.processor, self.model = None, None
        try:
            self.processor, self.model = _load(blip_model)
        except Exception:
            # fallback to base model
            base_name = "Salesforce/blip-image-captioning-base"
            self.processor, self.model = _load(base_name)

    def caption(self, image_path: str) -> str:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(img, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=32)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()
