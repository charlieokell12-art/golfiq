# GolfIQ Content Studio

A local VS Code content-generation workspace for TikTok, Instagram Reels and YouTube Shorts.

## What works now

- One prompt box for image or 15-second video jobs.
- 9:16 portrait workflow.
- 4K image export at 2160x3840.
- 4K H.264 video export with 24/30/48/60 fps choices.
- Smooth pan/zoom fallback videos that work without a generative GPU.
- Popular meme-template search using Imgflip's public catalogue (no API key required).
- Custom meme/image upload or direct-URL import.
- Meme overlay into rendered video.
- Local ComfyUI adapter: point the app at any API-format workflow JSON containing `__PROMPT__` placeholders.
- Model-agnostic design: Flux/SDXL image workflows and Wan/Hunyuan-style video workflows can be swapped without changing the UI.

## Important 4K note

Generating *natively* at 2160x3840 is wasteful and requires a very large GPU. The practical workflow is to generate at a model-friendly resolution, then upscale/render to 4K for social export. This is how Content Studio is designed.

## Run from VS Code

Install Python 3.11+ and FFmpeg, then:

```bash
cd content-studio
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The browser opens locally (normally `http://localhost:8501`).

## Free mode

Choose **Free fallback**. It creates a GolfIQ-styled vertical still from the prompt, then turns it into a smooth 15-second 4K video using FFmpeg. This is useful for testing the editor and for text/meme-led content without paying for generation.

## Free local AI mode

Install ComfyUI separately and start it with API access. Export an **API-format** workflow JSON from ComfyUI and put `__PROMPT__` anywhere the prompt text should be inserted. In Content Studio choose **Local ComfyUI**, enter the workflow path, and press Generate.

Recommended direction:

- Images: Flux or SDXL-class workflow.
- Video: a ComfyUI-compatible open video model such as Wan/Hunyuan-class workflow that your hardware can run.
- Upscale/interpolation: keep generation moderate-resolution, then use the Content Studio/FFmpeg export stage. A later release can add Real-ESRGAN and RIFE adapters.

Local model weights can be many gigabytes and are intentionally not stored in GitHub.

## Hardware reality

True text-to-video generation is GPU-heavy. A normal laptop without a suitable NVIDIA GPU can still run the UI, meme tools, editing, and 4K export, but it will not generate sophisticated moving AI footage locally at useful speed. The provider layer is deliberately replaceable so a free GPU host or future API can be connected later.

## Meme/copyright policy

The built-in search finds popular meme templates. Availability does not automatically grant commercial rights to every image. For GolfIQ advertising, prefer templates/media you have permission to reuse, public-domain/Creative-Commons sources, or original recreations. Users can also upload their own licensed assets.

## Viral-content research

The studio should learn *patterns*, not copy videos. Current research principles for GolfIQ shorts:

- Hook immediately, usually in the first 1–2 seconds.
- Keep visual movement active behind captions.
- Relatable golf situations and humour are more natural than a constant product pitch.
- Use GolfIQ as the payoff/solution rather than the entire subject.
- Fast cuts, captions, range/shot sounds and visual progress keep attention.
- End with a question/challenge that invites comments.
- Build repeatable formats (POV, challenge, myth, relatable fail, improvement, build-in-public) so one idea can produce many variants.

For bulk trend research, add a manifest of public video URLs/metadata and analyse hooks, length, pacing, captions and format. Do not download/reuse copyrighted source video unless its licence permits that use.
