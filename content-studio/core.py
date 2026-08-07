from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


@dataclass
class RenderResult:
    path: Path
    kind: str
    note: str


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("FFmpeg is required. Install it and make sure `ffmpeg` is on PATH.")
    return exe


def popular_memes() -> list[dict]:
    """Fetch Imgflip's public popular meme-template catalogue (no API key required)."""
    r = requests.get("https://api.imgflip.com/get_memes", timeout=20)
    r.raise_for_status()
    payload = r.json()
    if not payload.get("success"):
        raise RuntimeError("Imgflip meme catalogue request failed")
    return payload.get("data", {}).get("memes", [])


def search_memes(query: str, limit: int = 20) -> list[dict]:
    q = query.strip().lower()
    items = popular_memes()
    if q:
        items = [m for m in items if q in m.get("name", "").lower()]
    return items[:limit]


def download_url(url: str, suffix: str = ".jpg") -> Path:
    target = OUTPUTS / f"asset-{uuid.uuid4().hex}{suffix}"
    req = urllib.request.Request(url, headers={"User-Agent": "GolfIQ-Content-Studio/0.1"})
    with urllib.request.urlopen(req, timeout=30) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)
    return target


def make_prompt_card(prompt: str, title: str = "GolfIQ") -> Path:
    """Zero-cost fallback image so the editor remains usable without an AI backend."""
    w, h = 1080, 1920
    image = Image.new("RGB", (w, h), (8, 28, 20))
    draw = ImageDraw.Draw(image)
    try:
        font_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 86)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 50)
    except Exception:
        font_big = font_small = ImageFont.load_default()
    draw.text((72, 110), title, fill=(225, 191, 108), font=font_big)
    words = prompt.split()
    lines, line = [], []
    for word in words:
        line.append(word)
        if len(" ".join(line)) > 26:
            lines.append(" ".join(line[:-1]))
            line = [word]
    if line:
        lines.append(" ".join(line))
    y = 430
    for text in lines[:12]:
        draw.text((72, y), text, fill=(246, 246, 242), font=font_small)
        y += 72
    draw.text((72, 1710), "9:16 • GolfIQ Content Studio", fill=(160, 180, 166), font=font_small)
    path = OUTPUTS / f"prompt-{uuid.uuid4().hex}.png"
    image.save(path, quality=95)
    return path


def upscale_image_4k(source: Path) -> Path:
    """Upscale/crop to TikTok-native 4K portrait: 2160x3840."""
    out = OUTPUTS / f"4k-{uuid.uuid4().hex}.png"
    with Image.open(source).convert("RGB") as im:
        target_ratio = 2160 / 3840
        ratio = im.width / im.height
        if ratio > target_ratio:
            new_w = int(im.height * target_ratio)
            left = (im.width - new_w) // 2
            im = im.crop((left, 0, left + new_w, im.height))
        else:
            new_h = int(im.width / target_ratio)
            top = max(0, (im.height - new_h) // 2)
            im = im.crop((0, top, im.width, top + new_h))
        im = im.resize((2160, 3840), Image.Resampling.LANCZOS)
        im.save(out, quality=96)
    return out


def animate_image_to_4k_video(source: Path, duration: int = 15, fps: int = 60) -> Path:
    """Create smooth active motion from a still with zoom/pan and export 4K portrait H.264."""
    out = OUTPUTS / f"video-{uuid.uuid4().hex}.mp4"
    frames = max(1, duration * fps)
    vf = (
        "scale=2304:4096:force_original_aspect_ratio=increase,"
        "crop=2160:3840,"
        f"zoompan=z='min(zoom+0.00045,1.10)':x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':d={frames}:s=2160x3840:fps={fps},"
        "format=yuv420p"
    )
    cmd = [ffmpeg(), "-y", "-loop", "1", "-i", str(source), "-vf", vf,
           "-t", str(duration), "-r", str(fps), "-c:v", "libx264", "-preset", "medium",
           "-crf", "18", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out


def overlay_meme(base_video: Path, meme_image: Path, start_s: float = 2.0, duration_s: float = 2.5) -> Path:
    """Overlay a meme template briefly in the lower third."""
    out = OUTPUTS / f"meme-{uuid.uuid4().hex}.mp4"
    end_s = start_s + duration_s
    filter_complex = (
        "[1:v]scale=900:-1[mm];"
        f"[0:v][mm]overlay=(W-w)/2:H-h-260:enable='between(t,{start_s},{end_s})'"
    )
    cmd = [ffmpeg(), "-y", "-i", str(base_video), "-loop", "1", "-i", str(meme_image),
           "-filter_complex", filter_complex, "-c:v", "libx264", "-crf", "18",
           "-preset", "medium", "-c:a", "copy", "-shortest", str(out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out


def comfyui_generate(prompt: str, mode: str, workflow_path: Path, server: str = "http://127.0.0.1:8188") -> RenderResult:
    """Queue a user-supplied ComfyUI API workflow template.

    The workflow JSON may include __PROMPT__ placeholders in string values. This keeps
    Content Studio model-agnostic: SDXL/Flux/Wan/Hunyuan workflows can all be swapped in.
    """
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

    def replace(value):
        if isinstance(value, str):
            return value.replace("__PROMPT__", prompt)
        if isinstance(value, list):
            return [replace(v) for v in value]
        if isinstance(value, dict):
            return {k: replace(v) for k, v in value.items()}
        return value

    workflow = replace(workflow)
    client_id = uuid.uuid4().hex
    response = requests.post(f"{server.rstrip('/')}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=30)
    response.raise_for_status()
    prompt_id = response.json()["prompt_id"]

    deadline = time.time() + 60 * 45
    while time.time() < deadline:
        history = requests.get(f"{server.rstrip('/')}/history/{prompt_id}", timeout=20).json()
        record = history.get(prompt_id)
        if record:
            outputs = record.get("outputs", {})
            for node in outputs.values():
                for key in ("images", "gifs"):
                    for item in node.get(key, []):
                        filename = item.get("filename")
                        if filename:
                            params = {"filename": filename, "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")}
                            raw = requests.get(f"{server.rstrip('/')}/view", params=params, timeout=60)
                            raw.raise_for_status()
                            suffix = Path(filename).suffix or (".mp4" if mode == "video" else ".png")
                            target = OUTPUTS / f"comfy-{uuid.uuid4().hex}{suffix}"
                            target.write_bytes(raw.content)
                            return RenderResult(target, mode, "Generated locally through ComfyUI")
        time.sleep(2)
    raise TimeoutError("ComfyUI generation timed out after 45 minutes")
