import json
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional, Iterable, Dict, Any


def load_config(config_path="config.yaml") -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def list_images(paths: Iterable[str]) -> List[str]:
    out = []
    for p in paths:
        pth = Path(p)
        if pth.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
                out += [str(q) for q in pth.glob(ext)]
        else:
            # support simple globs too
            if any(ch in p for ch in ["*", "?", "["]):
                out += [str(q) for q in Path().glob(p)]
            elif pth.exists():
                out.append(str(pth))
    return sorted(out)


# ---------- ComfyUI output pickup ----------

def find_comfy_output_dir(cfg: Dict[str, Any]) -> Optional[Path]:
    # Respect explicit config if present
    out_override = cfg.get("comfy_output_dir")
    if out_override:
        p = Path(out_override)
        if p.exists():
            return p

    # Common defaults
    candidates = [
        Path("output"),
        Path("outputs"),
        Path("..") / "output",
        Path("..") / "outputs",
        Path("C:/ComfyUI/ComfyUI/output"),
        Path.home() / "ComfyUI" / "output",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_recent_outputs(output_dir: Path, filename_prefix: str, since_ts: float, limit: int = 1) -> List[Path]:
    files = []
    if not output_dir.exists():
        return files
    for p in output_dir.glob(f"{filename_prefix}*"):
        try:
            if p.stat().st_mtime >= since_ts - 1.0:  # small slack
                files.append(p)
        except Exception:
            continue
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[:limit]


def copy_into_outputs(src: Path, dst_dir: Path) -> Path:
    ensure_dir(str(dst_dir))
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


# ---------- Checkpoint discovery (disk) ----------

def _possible_comfy_roots(cfg: Dict[str, Any]) -> List[Path]:
    roots = []
    # 1) explicit
    root_cfg = cfg.get("comfy_root")
    if root_cfg:
        roots.append(Path(root_cfg))

    # 2) working dir and parents
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        if (p / "models" / "checkpoints").exists():
            roots.append(p)
        if p.name.lower() == "comfyui" and (p / "output").exists():
            roots.append(p)

    # 3) common Windows locations
    roots += [
        Path("C:/ComfyUI/ComfyUI"),
        Path.home() / "ComfyUI" / "ComfyUI",
        Path.home() / "ComfyUI"
    ]
    # Dedup while preserving order
    seen, out = set(), []
    for r in roots:
        try:
            rp = r.resolve()
        except Exception:
            rp = r
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def discover_checkpoints_on_disk(cfg: Dict[str, Any]) -> List[str]:
    """Return names the way ComfyUI usually displays them:
       - subdir/name.safetensors  (forward slashes)
       - also include bare filename as a fallback
    """
    names = []
    for root in _possible_comfy_roots(cfg):
        ck_dir = root / "models" / "checkpoints"
        if not ck_dir.exists():
            continue
        for p in ck_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".safetensors", ".ckpt"):
                # relative to checkpoints dir, forward slashes
                rel = p.relative_to(ck_dir).as_posix()
                names.append(rel)
                # add bare filename too
                names.append(p.name)
    # de-dup, keep order
    out, seen = [], set()
    for n in names:
        if n not in seen:
            out.append(n); seen.add(n)
    return out


# ---------- Sampler/scheduler normalization ----------

# Minimal mapping of common synonyms → Comfy sampler names
_SAMPLER_MAP = {
    "euler_a": "euler_ancestral",
    "euler-ancestral": "euler_ancestral",
    "euler ancestral": "euler_ancestral",
    "euler": "euler",
}

def _norm(x: str) -> str:
    return (x or "").strip().lower()

def normalize_sampler(requested: str, server_list: List[str]) -> str:
    req = _norm(requested)
    if not server_list:
        return _SAMPLER_MAP.get(req, requested)
    # exact or case-insensitive exact
    for s in server_list:
        if s == requested or _norm(s) == req:
            return s
    # mapped synonym
    mapped = _SAMPLER_MAP.get(req, requested)
    # endswith heuristic
    cand = [s for s in server_list if _norm(s).endswith(_norm(mapped))]
    if len(cand) == 1:
        return cand[0]
    return mapped

def normalize_scheduler(requested: str, server_list: List[str]) -> str:
    req = _norm(requested)
    if not server_list:
        return requested
    for s in server_list:
        if s == requested or _norm(s) == req:
            return s
    cand = [s for s in server_list if _norm(s).endswith(req)]
    if len(cand) == 1:
        return cand[0]
    return requested
