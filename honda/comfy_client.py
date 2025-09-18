# honda/comfy_client.py
from __future__ import annotations
import json
import pathlib
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


class ComfyClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8188,
        request_timeout_s: int = 60,
        comfy_output_dir: Optional[str] = None,
    ):
        self.base = f"http://{host}:{port}"
        self.request_timeout_s = request_timeout_s
        # If provided, used as fallback when /view can’t serve files
        self.comfy_output_dir = (
            pathlib.Path(comfy_output_dir).resolve()
            if comfy_output_dir
            else None
        )

    # ---------------- HTTP helpers ----------------

    def _get_json(self, path: str) -> Dict[str, Any]:
        url = f"{self.base}{path}"
        try:
            with urllib.request.urlopen(url, timeout=self.request_timeout_s) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GET {url} -> HTTP {e.code}: {detail}") from None

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_s) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"POST {url} -> HTTP {e.code}: {detail}") from None

    def _get_bytes(self, path: str) -> bytes:
        url = f"{self.base}{path}"
        with urllib.request.urlopen(url, timeout=self.request_timeout_s) as r:
            return r.read()

    # ---------------- Public API ----------------

    def queue_prompt(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a workflow graph to ComfyUI and receive {"prompt_id": "...", ...}
        """
        return self._post_json("/prompt", {"prompt": workflow})

    def history(self, prompt_id: str) -> Dict[str, Any]:
        return self._get_json(f"/history/{prompt_id}")

    def await_prompt(
        self,
        prompt_id: str,
        poll_interval_s: float = 0.35,
        timeout_s: float = 90,
    ) -> Dict[str, Any]:
        """
        Poll /history/{prompt_id} until completed or timeout.
        Returns the single history entry dict (what Comfy calls 'entry').
        """
        deadline = time.time() + timeout_s
        last = None
        while time.time() < deadline:
            hist = self.history(prompt_id)
            # /history/{id} returns a dict keyed by the prompt number
            if isinstance(hist, dict) and hist:
                # take the only/last entry
                _, entry = next(reversed(list(hist.items())))
                last = entry
                status = (entry.get("status") or {}).get("status_str", "")
                completed = (entry.get("status") or {}).get("completed", False)
                if completed or status == "success":
                    return entry
            time.sleep(poll_interval_s)
        # return whatever we last saw, for diagnostics
        return last or {}

    def download_image(
        self,
        filename: str,
        subfolder: str,
        img_type: str,
        dst_path: pathlib.Path,
    ) -> None:
        """
        Try Comfy's /view endpoint first; caller may catch HTTPError to fall back to disk.
        """
        q = urllib.parse.urlencode(
            {"filename": filename, "type": img_type, "subfolder": subfolder}
        )
        data = self._get_bytes("/view?" + q)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(data)

    def _disk_try(self, base: pathlib.Path, name: str) -> Optional[pathlib.Path]:
        """
        Return existing path if name exists under base (recurses a couple of common layouts).
        """
        # 1) direct child
        p = base / name
        if p.exists():
            return p
        # 2) Comfy's default "output" folder + flattened name
        #    (when SaveImage subfolder was blank)
        for sub in ["", "output", "user/output", "ComfyUI/output"]:
            pp = base / sub / name
            if pp.exists():
                return pp
        return None

    def download_all_outputs(
        self,
        entry: Dict[str, Any],
        dst_dir: pathlib.Path | str,
        *,
        forced_subfolder: Optional[str] = None,
        # NEW: optional base dir to search if /view 404s (kept name 'disk_fallback' to match caller)
        disk_fallback: Optional[str | pathlib.Path] = None,
    ) -> List[str]:
        """
        Download every image listed in the history 'entry' into dst_dir.
        Returns list of saved paths.

        - First tries /view (authoritative).
        - If /view returns 404 or similar, will search the disk under:
            1) 'disk_fallback' if provided, else
            2) self.comfy_output_dir if provided.
        """
        dst_dir = pathlib.Path(dst_dir)
        outputs = (entry.get("outputs") or {})
        images = []
        for node_id, node_out in outputs.items():
            for img in (node_out.get("images") or []):
                fname = img.get("filename")
                subfolder = img.get("subfolder") or ""
                img_type = img.get("type") or "output"

                # Optionally force a subfolder (helps when Comfy wrote to a known folder)
                sf = forced_subfolder if forced_subfolder is not None else subfolder

                target = dst_dir / fname
                try:
                    self.download_image(fname, sf, img_type, target)
                    images.append(str(target))
                    continue
                except urllib.error.HTTPError as e:
                    # Only fall back on 404; other codes should surface
                    if e.code != 404:
                        raise
                except Exception:
                    # unexpected networking failure -> fall back below
                    pass

                # Disk fallback
                base = (
                    pathlib.Path(disk_fallback).resolve()
                    if disk_fallback
                    else (self.comfy_output_dir or None)
                )
                if base is not None:
                    found = self._disk_try(base, fname)
                    if found is not None:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        data = found.read_bytes()
                        target.write_bytes(data)
                        images.append(str(target))
                        continue

                raise RuntimeError(
                    f"Could not fetch '{fname}' via /view (type='{img_type}', subfolder='{sf}')"
                    + (f" and disk fallback under '{base}'." if base else ".")
                )

        return images
