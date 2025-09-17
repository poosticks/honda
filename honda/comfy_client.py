import json
import time
import uuid
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode
import requests


class ComfyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188, timeout_s: int = 90):
        self.base = f"http://{host}:{port}"
        self.client_id = uuid.uuid4().hex
        self.timeout_s = timeout_s

    def _raise_for_status(self, resp: requests.Response):
        if 200 <= resp.status_code < 300:
            return
        msg = resp.text
        try:
            msg = json.dumps(resp.json())
        except Exception:
            pass
        raise RuntimeError(f"ComfyUI HTTP {resp.status_code}: {msg}")

    # -------------------- Discovery --------------------

    def get_checkpoints(self) -> List[str]:
        """
        Return checkpoint names exactly as ComfyUI expects in CheckpointLoaderSimple's ckpt_name.
        Tries /models/ckpt first; falls back to /object_info enum if available.
        """
        # 1) /models/ckpt (present on most builds)
        try:
            r = requests.get(f"{self.base}/models/ckpt", timeout=self.timeout_s)
            if 200 <= r.status_code < 300:
                data = r.json()
                if isinstance(data, list):
                    # items may be dicts or strings depending on build
                    out = []
                    for it in data:
                        if isinstance(it, str):
                            out.append(it)
                        elif isinstance(it, dict):
                            # common keys: "name", "filename", "display_name"
                            out.append(it.get("name") or it.get("display_name") or it.get("filename") or str(it))
                    return [x for x in out if x]
        except Exception:
            pass

        # 2) /object_info → CheckpointLoaderSimple enum of ckpt_name
        try:
            r = requests.get(f"{self.base}/object_info", timeout=self.timeout_s)
            self._raise_for_status(r)
            info = r.json() or {}
            node = info.get("CheckpointLoaderSimple") or {}
            itypes = node.get("input_types") or {}
            req = itypes.get("required") or {}
            ck = req.get("ckpt_name")
            if isinstance(ck, dict):
                choices = ck.get("choices") or ck.get("enum") or []
                if isinstance(choices, list):
                    return [str(x) for x in choices]
        except Exception:
            pass

        return []

    def get_samplers(self) -> List[str]:
        try:
            r = requests.get(f"{self.base}/samplers", timeout=self.timeout_s)
            if 200 <= r.status_code < 300:
                data = r.json()
                if isinstance(data, list):
                    if data and isinstance(data[0], dict):
                        return [d.get("name") or d.get("sampler_name") or str(d) for d in data]
                    return [str(x) for x in data]
        except Exception:
            pass
        return self._enum_from_object_info("KSampler", "sampler_name")

    def get_schedulers(self) -> List[str]:
        try:
            r = requests.get(f"{self.base}/schedulers", timeout=self.timeout_s)
            if 200 <= r.status_code < 300:
                data = r.json()
                if isinstance(data, list):
                    if data and isinstance(data[0], dict):
                        return [d.get("name") or str(d) for d in data]
                    return [str(x) for x in data]
        except Exception:
            pass
        return self._enum_from_object_info("KSampler", "scheduler")

    def _enum_from_object_info(self, node_name: str, field: str) -> List[str]:
        try:
            r = requests.get(f"{self.base}/object_info", timeout=self.timeout_s)
            self._raise_for_status(r)
            info = r.json() or {}
            node = info.get(node_name) or {}
            itypes = node.get("input_types") or {}
            req = itypes.get("required") or {}
            if isinstance(req.get(field), dict):
                choices = req[field].get("choices") or req[field].get("enum") or []
                if isinstance(choices, list):
                    return [str(x) for x in choices]
        except Exception:
            pass
        return []

    # -------------------- Submission --------------------

    def submit(self, workflow: dict) -> str:
        # Keep payload minimal; some extensions choke on extra_pnginfo
        payload = {"prompt": workflow, "client_id": self.client_id}
        r = requests.post(f"{self.base}/prompt", json=payload, timeout=self.timeout_s)
        self._raise_for_status(r)
        data = r.json()
        pid = data.get("prompt_id") or data.get("id") or data.get("name")
        if not pid:
            raise RuntimeError(f"Unexpected /prompt response: {data}")
        return pid

    @staticmethod
    def _outputs_have_images(obj: Dict[str, Any]) -> bool:
        outputs = obj.get("outputs", {}) or {}
        if not isinstance(outputs, dict) or not outputs:
            return False
        for _, out in outputs.items():
            imgs = (out or {}).get("images")
            if isinstance(imgs, list) and imgs:
                return True
        return False

    @staticmethod
    def _status_str(d: Dict[str, Any]) -> str:
        return ((d.get("status") or {}).get("status_str") or d.get("status_str") or "").lower()

    def wait_for_completion(self, prompt_id: str, poll_s: float = 1.0, max_wait_s: int = 600) -> dict:
        url = f"{self.base}/history/{prompt_id}"
        start = time.time()
        last = None
        FINISHED = {"success", "completed", "done", "finished", "executed"}
        FAILED = {"error", "canceled", "cancelled", "failed"}

        while True:
            r = requests.get(url, timeout=self.timeout_s)
            self._raise_for_status(r)
            data = r.json()

            def check(obj: Dict[str, Any]):
                if self._outputs_have_images(obj):
                    return obj, True
                status = self._status_str(obj)
                if status in FAILED:
                    raise RuntimeError(f"ComfyUI run failed: {obj}")
                if status in FINISHED:
                    return obj, True  # finished but empty → caller can disk-fallback
                return obj, False

            if isinstance(data, dict) and "history" in data and prompt_id in data["history"]:
                inner = data["history"][prompt_id] or {}
                last, done = check(inner)
                if done:
                    return inner
            elif isinstance(data, dict):
                last, done = check(data)
                if done:
                    return data

            if time.time() - start > max_wait_s:
                raise TimeoutError(f"Timed out waiting for prompt_id={prompt_id}; last={last}")
            time.sleep(poll_s)

    def get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        qs = urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        r = requests.get(f"{self.base}/view?{qs}", timeout=self.timeout_s)
        self._raise_for_status(r)
        return r.content
