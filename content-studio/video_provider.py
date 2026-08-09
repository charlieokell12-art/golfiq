from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import requests

from core import OUTPUTS


@dataclass
class GeneratedAsset:
    path: Path
    note: str


def _replace_tokens(value, replacements: dict[str, str]):
    if isinstance(value, str):
        for key, replacement in replacements.items():
            value = value.replace(key, replacement)
        return value
    if isinstance(value, list):
        return [_replace_tokens(v, replacements) for v in value]
    if isinstance(value, dict):
        return {k: _replace_tokens(v, replacements) for k, v in value.items()}
    return value


def _candidate_items(node: dict):
    # ComfyUI core image nodes normally return `images`. Popular video nodes often
    # expose `gifs`, `videos`, `files`, or filenames inside a nested object.
    for key in ("videos", "gifs", "images", "files"):
        items = node.get(key, [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("filename"):
                    yield item


def generate_video(
    prompt: str,
    workflow_path: Path,
    server: str,
    *,
    duration: float,
    fps: int,
    seed: int | None = None,
    timeout_minutes: int = 60,
) -> GeneratedAsset:
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    replacements = {
        "__PROMPT__": prompt,
        "__DURATION__": str(duration),
        "__FPS__": str(fps),
        "__FRAMES__": str(max(1, int(round(duration * fps)))),
        "__SEED__": str(seed if seed is not None else int(time.time() * 1000) % 2147483647),
    }
    workflow = _replace_tokens(workflow, replacements)

    base = server.rstrip("/")
    client_id = uuid.uuid4().hex
    queued = requests.post(f"{base}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=30)
    queued.raise_for_status()
    prompt_id = queued.json()["prompt_id"]

    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        response = requests.get(f"{base}/history/{prompt_id}", timeout=20)
        response.raise_for_status()
        record = response.json().get(prompt_id)
        if record:
            status = record.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI workflow failed: {status}")

            for node in record.get("outputs", {}).values():
                for item in _candidate_items(node):
                    filename = item["filename"]
                    params = {
                        "filename": filename,
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                    }
                    raw = requests.get(f"{base}/view", params=params, timeout=120)
                    raw.raise_for_status()
                    suffix = Path(filename).suffix.lower() or ".mp4"
                    target = OUTPUTS / f"comfy-video-{uuid.uuid4().hex}{suffix}"
                    target.write_bytes(raw.content)
                    return GeneratedAsset(target, "Generated as a live-motion scene through local ComfyUI")
        time.sleep(2)

    raise TimeoutError(f"ComfyUI scene generation exceeded {timeout_minutes} minutes")
