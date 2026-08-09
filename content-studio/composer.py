from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from core import OUTPUTS, ffmpeg, make_prompt_card
from director import StoryPlan
from video_provider import generate_video


@dataclass
class SceneRender:
    index: int
    path: Path
    duration: float
    note: str


def _normalise_scene(source: Path, duration: float, fps: int = 30) -> Path:
    """Normalise an arbitrary generated scene to a TikTok-friendly intermediate."""
    out = OUTPUTS / f"scene-{uuid.uuid4().hex}.mp4"
    suffix = source.suffix.lower()
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=%d,format=yuv420p" % fps

    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        cmd = [
            ffmpeg(), "-y", "-loop", "1", "-i", str(source), "-vf", vf,
            "-t", str(duration), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an", str(out),
        ]
    else:
        cmd = [
            ffmpeg(), "-y", "-i", str(source), "-vf", vf,
            "-t", str(duration), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an", str(out),
        ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out


def render_scenes(
    plan: StoryPlan,
    provider: str,
    workflow_path: Path | None,
    comfy_server: str,
    fps: int = 30,
) -> list[SceneRender]:
    renders: list[SceneRender] = []
    for scene in plan.scenes:
        if provider == "Local ComfyUI":
            if not workflow_path:
                raise RuntimeError("Local ComfyUI needs a video workflow JSON path.")
            result = generate_video(
                scene.generation_prompt,
                workflow_path,
                comfy_server,
                duration=scene.duration,
                fps=fps,
                # Holding a stable scene seed helps style consistency while prompts
                # preserve character/location continuity across the full story.
                seed=24681357 + scene.index,
            )
            raw = result.path
            note = result.note
        else:
            raw = make_prompt_card(
                f"SCENE {scene.index}: {scene.action}\nCAMERA: {scene.camera}\nDIALOGUE: {scene.dialogue}",
                title=f"GolfIQ Director · Scene {scene.index}",
            )
            note = "Storyboard preview (connect a local video model for live-action generation)"
        clip = _normalise_scene(raw, scene.duration, fps=fps)
        renders.append(SceneRender(scene.index, clip, scene.duration, note))
    return renders


def _write_concat_file(renders: list[SceneRender]) -> Path:
    manifest = OUTPUTS / f"concat-{uuid.uuid4().hex}.txt"
    lines = []
    for item in renders:
        safe = str(item.path.resolve()).replace("'", "'\\''")
        lines.append(f"file '{safe}'")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def stitch_and_export_4k(renders: list[SceneRender], fps: int = 30) -> Path:
    if not renders:
        raise RuntimeError("No scenes were rendered.")
    concat_file = _write_concat_file(renders)
    joined = OUTPUTS / f"joined-{uuid.uuid4().hex}.mp4"
    subprocess.run(
        [
            ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(joined),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    final = OUTPUTS / f"golfiq-director-4k-{uuid.uuid4().hex}.mp4"
    vf = f"scale=2160:3840:flags=lanczos,fps={fps},format=yuv420p"
    subprocess.run(
        [
            ffmpeg(), "-y", "-i", str(joined), "-vf", vf,
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-movflags", "+faststart", str(final),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return final


def save_plan(plan: StoryPlan) -> Path:
    path = OUTPUTS / f"story-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    return path
