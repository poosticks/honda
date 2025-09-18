# honda/metrics.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import torch
import torchvision.transforms as T

# Optional deps
try:
    import clip  # OpenAI CLIP
except Exception:
    clip = None

try:
    import lpips  # Perceptual metric
except Exception:
    lpips = None


class CompositeScorer:
    """
    Combines LPIPS (lower is better) and CLIP cosine (higher is better)
    into a single 'combined' score used for selection.

    combined = w_clip * clip_cos + (1 - w_clip) * (1 - min(1, lpips))

    You can extend with DINO/DreamSim later and fold into 'combined'.
    """

    def __init__(self, cfg: Dict[str, Any]):
        cfg = cfg or {}
        dev = cfg.get("device", "cpu")
        if isinstance(dev, str) and dev.startswith("cuda") and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.use_lpips = bool(cfg.get("use_lpips", True))
        self.use_clip = bool(cfg.get("use_clip", True))
        self.use_dino = bool(cfg.get("use_dino", False))          # reserved
        self.use_dreamsim = bool(cfg.get("use_dreamsim", False))  # reserved
        self.clip_model_name = cfg.get("clip_model", "ViT-B/32")
        self.resize = int(cfg.get("resize", 256))
        self.combine = (cfg.get("combine") or {"method": "weighted", "w_clip": 0.6})

        self._lpips = None
        self._clip = None
        self._clip_pre = None

        if self.use_lpips:
            if lpips is None:
                raise RuntimeError("LPIPS requested but package 'lpips' not installed.")
            self._lpips = lpips.LPIPS(net="alex").to(self.device).eval()

        if self.use_clip:
            if clip is None:
                raise RuntimeError("CLIP requested but package 'clip' not installed.")
            self._clip, self._clip_pre = clip.load(self.clip_model_name, device=self.device)
            self._clip.eval()

        self._pil_to_tensor = T.Compose([
            T.Resize(self.resize, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self.resize),
            T.ToTensor(),
        ])

    def _load_image_tensor(self, pil_img):
        # PIL.Image -> (1,3,H,W) on device
        t = self._pil_to_tensor(pil_img).unsqueeze(0).to(self.device)
        return t

    def score_pair(self, ref_img, cand_img) -> Dict[str, float]:
        """
        ref_img, cand_img are PIL.Image objects.
        Returns dict with lpips, clip_cos and combined.
        """
        with torch.no_grad():
            ref_t = self._load_image_tensor(ref_img)
            cand_t = self._load_image_tensor(cand_img)

            out: Dict[str, float] = {}

            if self.use_lpips:
                d = self._lpips(ref_t, cand_t)
                lp_val = float(d.detach().cpu().numpy().mean())
                out["lpips"] = lp_val
            else:
                out["lpips"] = 0.0

            if self.use_clip:
                # CLIP expects normalized preprocess; use official preprocess if available
                pre = self._clip_pre or (lambda x: x)
                ref_c = pre(ref_img).unsqueeze(0).to(self.device)
                cand_c = pre(cand_img).unsqueeze(0).to(self.device)
                text = None
                # Encode images, cosine sim
                ir = self._clip.encode_image(ref_c)
                ic = self._clip.encode_image(cand_c)
                ir = ir / ir.norm(dim=-1, keepdim=True)
                ic = ic / ic.norm(dim=-1, keepdim=True)
                clip_cos = float((ir @ ic.T).squeeze().clamp(-1, 1).detach().cpu().numpy())
                out["clip_cos"] = clip_cos
            else:
                out["clip_cos"] = 0.0

            # Combined
            lpips_sim = max(0.0, 1.0 - min(1.0, out["lpips"]))
            w = float(self.combine.get("w_clip", 0.6))
            out["combined"] = (w * out["clip_cos"]) + ((1.0 - w) * lpips_sim)

            return out
