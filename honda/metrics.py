import torch
import torchvision.transforms as T
from PIL import Image

# Load LPIPS model (AlexNet backbone by default)
import lpips
_lpips_model = None

# Load OpenAI CLIP model for image embeddings
try:
    import clip
    _clip_model, _clip_preprocess = None, None
except ImportError:
    clip = None
    _clip_model, _clip_preprocess = None, None

# Optionally, handle DINO or other models if needed (not using transformers here to avoid overhead)

# Optional DreamSim integration
try:
    from dreamsim import dreamsim as _dreamsim_fn
    _dreamsim_model = _dreamsim_fn()  # This might load a default DreamSim ensemble
except ImportError:
    _dreamsim_model = None

# Basic image preprocessing: resize to 224 and center-crop (for CLIP, DINO etc. standard input)
_clip_input_transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor()
])
# Normalization for CLIP (if using OpenAI CLIP ViT-B/32)
_clip_normalize = T.Normalize((0.48145466, 0.4578275, 0.40821073),
                              (0.26862954, 0.26130258, 0.27577711))

def _load_models(device="cuda"):
    global _lpips_model, _clip_model, _clip_preprocess
    dev = torch.device(device if torch.cuda.is_available() or device=="cpu" else "cpu")
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex').to(dev)  # AlexNet-based LPIPS model
    if clip is not None and _clip_model is None:
        _clip_model, _clip_preprocess = clip.load('ViT-B/32', device=dev)  # Load CLIP model
        # _clip_preprocess is a torchvision transform matching CLIP training (224x224 + normalize)
    return dev

def compute_lpips(img_path1: str, img_path2: str, device="cuda") -> float:
    """Compute LPIPS distance between two images (lower = more similar)."""
    dev = _load_models(device)
    # Load images
    img0 = Image.open(img_path1).convert("RGB")
    img1 = Image.open(img_path2).convert("RGB")
    # Resize both to same size (e.g., 256px shortest side) to avoid mismatch
    # (LPIPS expects same spatial dimensions)
    transform = T.Compose([
        T.Resize(256, max_size=256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(256),
        T.ToTensor()
    ])
    t0 = transform(img0).unsqueeze(0).to(dev)
    t1 = transform(img1).unsqueeze(0).to(dev)
    # Normalize to [-1,1] as LPIPS expects
    t0 = (t0 * 2 - 1)
    t1 = (t1 * 2 - 1)
    with torch.no_grad():
        dist = _lpips_model(t0, t1)  # LPIPS returns a tensor distance
    return float(dist.item())

def compute_clip_similarity(img_path1: str, img_path2: str, device="cuda") -> float:
    """
    Compute cosine similarity between CLIP image embeddings of two images (higher = more similar).
    """
    if clip is None:
        raise RuntimeError("OpenAI CLIP not available. Install openai-clip or check dependencies.")
    dev = _load_models(device)
    img1 = _clip_preprocess(Image.open(img_path1).convert("RGB")).unsqueeze(0).to(dev)
    img2 = _clip_preprocess(Image.open(img_path2).convert("RGB")).unsqueeze(0).to(dev)
    with torch.no_grad():
        emb1 = _clip_model.encode_image(img1)
        emb2 = _clip_model.encode_image(img2)
        emb1 /= emb1.norm(dim=-1, keepdim=True)
        emb2 /= emb2.norm(dim=-1, keepdim=True)
        sim = (emb1 * emb2).sum(dim=-1)  # cosine similarity since vectors are normalized
    return float(sim.item())

def compute_composite_score(img_path: str, ref_path: str, use_dream=False, device="cuda"):
    """Compute combined similarity score for one candidate image vs reference."""
    # Lower LPIPS = more similar, so use (1 - LPIPS) as similarity contribution.
    lpips_val = compute_lpips(ref_path, img_path, device=device)
    lpips_sim = max(0.0, 1.0 - lpips_val)  # clamp to [0,1]
    try:
        clip_sim = compute_clip_similarity(ref_path, img_path, device=device)
    except Exception as e:
        clip_sim = 0.0
    # clip_sim is already cosine similarity in [-1,1]; assume most will be in [0,1] for unrelated to identical.
    clip_sim_norm = (clip_sim + 1) / 2.0  # normalize to [0,1]
    # Optionally, DreamSim (if available and use_dream=True)
    dream_sim = None
    if use_dream and _dreamsim_model is not None:
        # DreamSim usage: model might accept PIL or tensors. We'll assume PIL for simplicity.
        try:
            im_ref = Image.open(ref_path).convert("RGB")
            im_cand = Image.open(img_path).convert("RGB")
            # dreamsim returns a distance; smaller = more similar.
            d = _dreamsim_model(im_ref, im_cand)
            if isinstance(d, torch.Tensor):
                d = d.item()
            # Convert distance to similarity (roughly invert; DreamSim distances are ~0 to 2 for very different)
            dream_sim = max(0.0, 1.0 - (d / 2.0))
        except Exception as e:
            dream_sim = None
    # Composite: We give equal weight to LPIPS-sim and CLIP-sim for now, and optionally incorporate DreamSim.
    if dream_sim is not None:
        # If DreamSim is used, we can average all three.
        composite = (lpips_sim + clip_sim_norm + dream_sim) / 3.0
    else:
        composite = (lpips_sim + clip_sim_norm) / 2.0
    return {
        "lpips": lpips_val,
        "lpips_sim": lpips_sim,
        "clip_sim": clip_sim,
        "dream_sim": dream_sim,
        "score": composite
    }
